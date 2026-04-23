#!/usr/bin/env python3
"""
Generative listwise reranker evaluation for CodeGRIP.

Instead of pointwise yes/no classification, this uses the base Qwen2.5-7B-Instruct
(ZERO-SHOT, no LoRA) to generatively rank top-20 candidate files given a bug report.

Purpose: test whether path prior is architecture-general — i.e., does a completely
different scoring mechanism (generative ranking) also rely on file paths?

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_generative_reranker.py --gpu_id 0 --perturb none
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_generative_reranker.py --gpu_id 0 --perturb shuffle_filenames
"""

import os
import sys
import json
import re
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
TEST_DATA = os.path.join(BASE_DIR, "data", "grepo_text", "grepo_test.jsonl")
CANDIDATES_DATA = os.path.join(BASE_DIR, "data", "rankft", "merged_bm25_exp6_candidates.jsonl")
PERTURB_TEST = {
    "shuffle_filenames": os.path.join(BASE_DIR, "experiments", "path_perturb_shuffle_filenames", "test.jsonl"),
    "shuffle_dirs": os.path.join(BASE_DIR, "experiments", "path_perturb_shuffle_dirs", "test.jsonl"),
}
PERTURB_CANDIDATES = {
    "shuffle_filenames": os.path.join(BASE_DIR, "experiments", "path_perturb_shuffle_filenames", "bm25_candidates.jsonl"),
    "shuffle_dirs": os.path.join(BASE_DIR, "experiments", "path_perturb_shuffle_dirs", "bm25_candidates.jsonl"),
}

MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"

TOP_K_CANDIDATES = 20  # default; can override via --top_k
MAX_EXAMPLES = 99999  # default: all examples
MAX_NEW_TOKENS = 500  # increased for top-20 ranking output


def build_ranking_prompt(issue_text: str, candidates: List[str]) -> str:
    """Build a listwise ranking prompt."""
    cand_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        "Given this bug report, rank the following candidate files from most to least "
        "likely to contain the bug. Output ONLY the ranking as a numbered list.\n\n"
        f"Bug Report: {issue_text}\n\n"
        f"Candidate files:\n{cand_list}\n\n"
        "Ranking (most likely first):"
    )
    return prompt


def _try_add_by_number(ref_1based: int, num_candidates: int, ranked_indices: list) -> bool:
    """Try to add a 1-based number reference as a 0-based index."""
    idx = ref_1based - 1
    if 0 <= idx < num_candidates and idx not in ranked_indices:
        ranked_indices.append(idx)
        return True
    return False


def _try_add_by_path(content: str, candidates: List[str], ranked_indices: list) -> bool:
    """Try to match content against candidate paths."""
    content = content.strip().rstrip(",").strip()
    for idx, cand in enumerate(candidates):
        if idx in ranked_indices:
            continue
        if content == cand or content.endswith(cand) or cand.endswith(content):
            ranked_indices.append(idx)
            return True
    # Fuzzy: basename match
    content_base = os.path.basename(content.rstrip("/"))
    if content_base:
        for idx, cand in enumerate(candidates):
            if idx in ranked_indices:
                continue
            if os.path.basename(cand) == content_base:
                ranked_indices.append(idx)
                return True
    return False


def parse_ranking(output: str, candidates: List[str], num_candidates: int) -> List[int]:
    """
    Parse model output to extract a ranking of candidate indices.
    Returns list of 0-based indices in predicted rank order.
    Falls back to original order [0, 1, ..., N-1] if unparseable.

    Handles formats:
      - "2, 3, 4, 6, 8, 1, 5, 7, 9, 10"  (comma-separated numbers)
      - "4\n5\n9\n10\n..."                 (bare numbers per line)
      - "1. path/to/file.py\n2. ..."       (numbered list with paths)
      - "1. 5\n2. 3\n..."                  (numbered list with number refs)
    """
    output = output.strip()
    ranked_indices = []

    # Strategy 1: comma or space-separated numbers on a single line
    first_line = output.split("\n")[0].strip()
    if "," in first_line:
        tokens = first_line.split(",")
    elif re.match(r"^[\d\s]+$", first_line) and len(first_line.split()) >= 2:
        tokens = first_line.split()
    else:
        tokens = None

    if tokens is not None:
        for tok in tokens:
            tok = tok.strip()
            if re.match(r"^\d+$", tok):
                _try_add_by_number(int(tok), num_candidates, ranked_indices)
            else:
                _try_add_by_path(tok, candidates, ranked_indices)
        if len(ranked_indices) >= 2:
            for i in range(num_candidates):
                if i not in ranked_indices:
                    ranked_indices.append(i)
            return ranked_indices

    # Strategy 2: line-by-line parsing
    ranked_indices = []
    lines = output.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # "N. <something>" or "N) <something>"
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m:
            content = m.group(1).strip()
            # Try as path first
            if not _try_add_by_path(content, candidates, ranked_indices):
                # Try as number reference
                num_m = re.match(r"^(\d+)$", content)
                if num_m:
                    _try_add_by_number(int(num_m.group(1)), num_candidates, ranked_indices)
            continue

        # Bare number
        if re.match(r"^\d+$", line):
            _try_add_by_number(int(line), num_candidates, ranked_indices)
            continue

        # Bare path
        _try_add_by_path(line, candidates, ranked_indices)

    # Pad with remaining in original order
    if len(ranked_indices) >= 1:
        for i in range(num_candidates):
            if i not in ranked_indices:
                ranked_indices.append(i)
        return ranked_indices

    # Complete fallback
    return list(range(num_candidates))


def load_data(perturb: str, max_examples: int = MAX_EXAMPLES, top_k: int = TOP_K_CANDIDATES) -> List[dict]:
    """Load test data and candidates, return merged records."""
    if perturb in PERTURB_TEST:
        test_path = PERTURB_TEST[perturb]
        cand_path = PERTURB_CANDIDATES[perturb]
    else:
        test_path = TEST_DATA
        cand_path = CANDIDATES_DATA

    # Load test data
    test_data = {}
    with open(test_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            test_data[key] = rec

    # Load candidates
    cand_data = {}
    with open(cand_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            cand_data[key] = rec["candidates"]

    # Merge
    results = []
    for key, test_rec in test_data.items():
        if key not in cand_data:
            continue
        candidates = cand_data[key][:top_k]
        if len(candidates) == 0:
            continue
        if perturb in ("shuffle_filenames", "shuffle_dirs"):
            # For perturbed: changed_files are already perturbed in test.jsonl
            gt_files = set(test_rec.get("changed_py_files", test_rec.get("changed_files", [])))
        else:
            gt_files = set(test_rec.get("changed_py_files", test_rec.get("changed_files", [])))
        results.append({
            "repo": test_rec["repo"],
            "issue_id": test_rec["issue_id"],
            "issue_text": test_rec["issue_text"],
            "candidates": candidates,
            "gt_files": gt_files,
        })

    return results[:max_examples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--perturb", choices=["none", "shuffle_filenames", "shuffle_dirs"], default="none")
    parser.add_argument("--max_examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--top_k", type=int, default=TOP_K_CANDIDATES,
                        help="Number of candidates per example for listwise ranking")
    args = parser.parse_args()

    top_k = args.top_k
    max_examples = args.max_examples

    device = f"cuda:{args.gpu_id}"

    print(f"=== Generative Listwise Reranker Evaluation ===")
    print(f"Perturbation: {args.perturb}")
    print(f"Model: {MODEL_PATH} (ZERO-SHOT, no LoRA)")
    print(f"Top-K candidates: {top_k}")
    print(f"Max examples: {max_examples}")
    print(f"Device: {device}")
    print()

    # Load data
    data = load_data(args.perturb, max_examples, top_k)
    print(f"Loaded {len(data)} examples")

    # Load model in 4-bit
    print("Loading model in 4-bit...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    torch.cuda.empty_cache()

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()
    sys.stdout.flush()

    # Evaluate
    hits_at_1 = 0
    hits_at_5 = 0
    total = 0
    parse_failures = 0

    for i, example in enumerate(data):
        issue_text = example["issue_text"]
        candidates = example["candidates"]
        gt_files = example["gt_files"]
        num_cands = len(candidates)

        # Build prompt
        prompt = build_ranking_prompt(issue_text, candidates)

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy
                temperature=None,
                top_p=None,
            )

        # Decode generated tokens only
        gen_tokens = outputs[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Parse ranking
        ranking = parse_ranking(gen_text, candidates, num_cands)

        # Check if ranking is just the fallback
        is_fallback = (ranking == list(range(num_cands)))
        if is_fallback:
            parse_failures += 1

        # Debug: print first 3 examples
        if i < 3:
            print(f"  [DEBUG {i}] gen_text (first 200 chars): {gen_text[:200]!r}")
            print(f"  [DEBUG {i}] parsed ranking[:5]: {ranking[:5]} fallback={is_fallback}")
            sys.stdout.flush()

        # Compute metrics
        ranked_candidates = [candidates[idx] for idx in ranking]

        # R@1: any GT file in top 1?
        if gt_files & set(ranked_candidates[:1]):
            hits_at_1 += 1
        # R@5: any GT file in top 5?
        if gt_files & set(ranked_candidates[:5]):
            hits_at_5 += 1
        total += 1

        if (i + 1) % 10 == 0 or i == 0:
            r1 = hits_at_1 / total * 100
            r5 = hits_at_5 / total * 100
            print(f"[{i+1}/{len(data)}] R@1={r1:.2f}% R@5={r5:.2f}% "
                  f"parse_fail={parse_failures}/{total} "
                  f"({parse_failures/total*100:.1f}%)")
            sys.stdout.flush()

    # Final results
    print()
    print("=" * 60)
    print(f"GENERATIVE LISTWISE RERANKER — perturb={args.perturb}")
    print(f"Model: Qwen2.5-7B-Instruct (ZERO-SHOT, no LoRA)")
    print(f"Top-{top_k} candidates, {total} test examples")
    print(f"Parse failures (fallback to original order): {parse_failures}/{total} ({parse_failures/total*100:.1f}%)")
    print(f"  R@1  = {hits_at_1/total*100:.2f}%")
    print(f"  R@5  = {hits_at_5/total*100:.2f}%")
    print("=" * 60)

    # Save results
    results_dir = os.path.join(BASE_DIR, "experiments", "generative_reranker")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{args.perturb}.json")
    with open(results_file, "w") as f:
        json.dump({
            "perturb": args.perturb,
            "model": MODEL_PATH,
            "top_k": top_k,
            "num_examples": total,
            "parse_failures": parse_failures,
            "R@1": hits_at_1 / total * 100,
            "R@5": hits_at_5 / total * 100,
            "hits_at_1": hits_at_1,
            "hits_at_5": hits_at_5,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
