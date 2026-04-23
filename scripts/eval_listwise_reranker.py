#!/usr/bin/env python3
"""
Long-context listwise code-aware reranker for CodeGRIP.

Unlike the pointwise Yes/No baseline that scores each candidate independently,
this model sees TOP-20 candidates WITH THEIR CODE CONTENT at once and directly
outputs a ranking. This is the strongest possible zero-shot code baseline.

Three evaluation modes:
  full       — real paths + code content  (baseline)
  code_only  — hashed paths + code content (does code help without path info?)
  path_only  — real paths, no code        (does code add value?)

If code_only >> random and path_only ≈ full → code adds nothing even in listwise.
If code_only > path_only → code IS useful in listwise but not in pointwise.

Design:
  - TOP-20 candidates from BM25+graph pool
  - 100 lines of code per file (~400 tokens each)
  - Total prompt: ~8K tokens for 20 files → fits in 8K context
  - Qwen2.5-7B-Instruct in 4-bit (zero-shot, no LoRA)
  - Greedy decoding, temperature=0
  - Parse output ranking, fallback to original order if unparseable

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_listwise_reranker.py --mode full
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_listwise_reranker.py --mode code_only
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_listwise_reranker.py --mode path_only

Runtime estimate: ~2-3 hours for 300 examples on 1x RTX 4090 (single-example generation).
Memory estimate: ~5-6 GB VRAM (4-bit quantized 7B model, up to ~8K token prompts).
"""

import os
import sys
import json
import re
import hashlib
import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Paths ===
TEST_DATA = os.path.join(BASE_DIR, "data", "grepo_text", "grepo_test.jsonl")
CANDIDATES_DATA = os.path.join(BASE_DIR, "data", "rankft", "merged_bm25_exp6_candidates.jsonl")
REPOS_DIR = os.path.join(BASE_DIR, "data", "repos")
MODEL_PATH = "/data/shuyang/models/Qwen2.5-7B-Instruct"

# === Constants ===
TOP_K_CANDIDATES = 20
MAX_CODE_LINES = 100
MAX_NEW_TOKENS = 400  # enough for a 20-item numbered list
MAX_PROMPT_TOKENS = 7500  # leave room within 8K context


def hash_path(path: str) -> str:
    """Deterministic hash to anonymize file paths while keeping them distinguishable."""
    h = hashlib.md5(path.encode()).hexdigest()[:8]
    # Preserve extension so model knows it's a Python file
    _, ext = os.path.splitext(path)
    return f"file_{h}{ext}"


def read_file_lines(repo: str, filepath: str, max_lines: int = MAX_CODE_LINES) -> Optional[str]:
    """Read up to max_lines from a file in the repo checkout.

    Returns None if file not found or unreadable.
    """
    full_path = os.path.join(REPOS_DIR, repo, filepath)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines) if lines else None
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None


def build_prompt_full(issue_text: str, candidates: List[str],
                      code_contents: List[Optional[str]]) -> str:
    """Mode=full: real paths + code content."""
    parts = [
        "Given this bug report, rank the following files from most to least likely "
        "to contain the bug. Consider the code content carefully.\n",
        f"Bug Report: {issue_text}\n",
    ]
    for i, (path, code) in enumerate(zip(candidates, code_contents)):
        parts.append(f"=== File {i+1}: {path} ===")
        if code:
            parts.append(code)
        else:
            parts.append("(file content unavailable)")
        parts.append("")  # blank line separator

    parts.append("Ranking (most likely first, by file number):")
    return "\n".join(parts)


def build_prompt_code_only(issue_text: str, candidates: List[str],
                           code_contents: List[Optional[str]]) -> str:
    """Mode=code_only: hashed paths + code content.
    Tests whether code content alone (without path signal) is useful."""
    parts = [
        "Given this bug report, rank the following files from most to least likely "
        "to contain the bug. Consider the code content carefully.\n",
        f"Bug Report: {issue_text}\n",
    ]
    for i, (path, code) in enumerate(zip(candidates, code_contents)):
        hashed = hash_path(path)
        parts.append(f"=== File {i+1}: {hashed} ===")
        if code:
            parts.append(code)
        else:
            parts.append("(file content unavailable)")
        parts.append("")

    parts.append("Ranking (most likely first, by file number):")
    return "\n".join(parts)


def build_prompt_path_only(issue_text: str, candidates: List[str],
                           code_contents: List[Optional[str]]) -> str:
    """Mode=path_only: real paths, no code content.
    Identical to the existing generative reranker but with 20 candidates."""
    cand_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        "Given this bug report, rank the following candidate files from most to least "
        "likely to contain the bug. Output ONLY the ranking as a numbered list.\n\n"
        f"Bug Report: {issue_text}\n\n"
        f"Candidate files:\n{cand_list}\n\n"
        "Ranking (most likely first, by file number):"
    )
    return prompt


PROMPT_BUILDERS = {
    "full": build_prompt_full,
    "code_only": build_prompt_code_only,
    "path_only": build_prompt_path_only,
}


# === Ranking parser (reused logic from eval_generative_reranker.py) ===

def _try_add_by_number(ref_1based: int, num_candidates: int, ranked_indices: list) -> bool:
    """Try to add a 1-based number reference as a 0-based index."""
    idx = ref_1based - 1
    if 0 <= idx < num_candidates and idx not in ranked_indices:
        ranked_indices.append(idx)
        return True
    return False


def parse_ranking(output: str, num_candidates: int) -> Tuple[List[int], bool]:
    """Parse model output to extract ranking of candidate indices (0-based).

    Returns (ranking, is_fallback).
    Handles formats:
      - "2, 3, 4, 6, 8, 1, 5, 7, 9, 10"  (comma-separated numbers)
      - "4\\n5\\n9\\n10\\n..."              (bare numbers per line)
      - "1. 5\\n2. 3\\n..."                (numbered list with number refs)
    """
    output = output.strip()
    ranked_indices = []

    # Strategy 1: comma or space-separated numbers on first line
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
        if len(ranked_indices) >= 2:
            for i in range(num_candidates):
                if i not in ranked_indices:
                    ranked_indices.append(i)
            return ranked_indices, False

    # Strategy 2: line-by-line
    ranked_indices = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue

        # "N. <something>" or "N) <something>" — extract the number ref
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m:
            content = m.group(1).strip()
            # Try as number reference (file number from prompt)
            num_m = re.match(r"^(?:File\s+)?(\d+)", content)
            if num_m:
                _try_add_by_number(int(num_m.group(1)), num_candidates, ranked_indices)
            continue

        # Bare number
        if re.match(r"^\d+$", line):
            _try_add_by_number(int(line), num_candidates, ranked_indices)
            continue

    # Pad with remaining in original order
    if len(ranked_indices) >= 1:
        for i in range(num_candidates):
            if i not in ranked_indices:
                ranked_indices.append(i)
        return ranked_indices, False

    # Complete fallback
    return list(range(num_candidates)), True


def load_data(max_examples: int, top_k: int = TOP_K_CANDIDATES) -> List[dict]:
    """Load test data and candidates, return merged records."""
    # Load test data
    test_data = {}
    with open(TEST_DATA) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            test_data[key] = rec

    # Load candidates
    cand_data = {}
    with open(CANDIDATES_DATA) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            cand_data[key] = rec["candidates"]

    # Merge — deterministic ordering via sorted keys
    results = []
    for key in sorted(test_data.keys()):
        if key not in cand_data:
            continue
        test_rec = test_data[key]
        candidates = cand_data[key][:top_k]
        if len(candidates) == 0:
            continue
        gt_files = set(test_rec.get("changed_py_files", test_rec.get("changed_files", [])))
        if not gt_files:
            continue
        results.append({
            "repo": test_rec["repo"],
            "issue_id": test_rec["issue_id"],
            "issue_text": test_rec["issue_text"],
            "candidates": candidates,
            "gt_files": gt_files,
        })

    # Deterministic subset
    return results[:max_examples]


def truncate_prompt(prompt: str, tokenizer, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """If prompt exceeds max_tokens, truncate code blocks while preserving the ranking instruction at the end."""
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    if token_count <= max_tokens:
        return prompt
    # Split into body (files) and suffix (ranking instruction)
    suffix_marker = "\nRanking (most likely first"
    idx = prompt.rfind(suffix_marker)
    if idx == -1:
        suffix_marker = "\nRanking"
        idx = prompt.rfind(suffix_marker)
    if idx > 0:
        body = prompt[:idx]
        suffix = prompt[idx:]
        suffix_tokens = len(tokenizer.encode(suffix, add_special_tokens=False))
        body_budget = max_tokens - suffix_tokens - 10
        body_tokens = tokenizer.encode(body, add_special_tokens=False)
        if len(body_tokens) > body_budget:
            body = tokenizer.decode(body_tokens[:body_budget])
        return body + suffix
    # Fallback: proportional truncation keeping last 200 chars
    ratio = max_tokens / token_count
    return prompt[:int(len(prompt) * ratio * 0.90)] + prompt[-200:]


def compute_metrics(ranked_candidates: List[str], gt_files: Set[str]) -> Dict[str, int]:
    """Compute hit@K for K in {1, 3, 5, 10, 20}."""
    hits = {}
    for k in [1, 3, 5, 10, 20]:
        top_k = set(ranked_candidates[:k])
        hits[f"hit@{k}"] = 1 if (gt_files & top_k) else 0
    return hits


def main():
    parser = argparse.ArgumentParser(
        description="Long-context listwise code-aware reranker evaluation"
    )
    parser.add_argument("--mode", choices=["full", "code_only", "path_only"],
                        required=True,
                        help="full=paths+code, code_only=hashed_paths+code, path_only=paths_no_code")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Logical GPU id (use CUDA_VISIBLE_DEVICES for physical)")
    parser.add_argument("--max_examples", type=int, default=300,
                        help="Max test examples to evaluate")
    parser.add_argument("--top_k", type=int, default=TOP_K_CANDIDATES,
                        help="Number of candidates per query (default 20)")
    parser.add_argument("--max_code_lines", type=int, default=MAX_CODE_LINES,
                        help="Max lines of code to include per file (default 100)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: experiments/listwise_reranker)")
    args = parser.parse_args()

    top_k = args.top_k
    max_code_lines = args.max_code_lines

    device = f"cuda:{args.gpu_id}"
    build_prompt = PROMPT_BUILDERS[args.mode]

    output_dir = args.output_dir or os.path.join(BASE_DIR, "experiments", "listwise_reranker")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("LISTWISE CODE-AWARE RERANKER EVALUATION")
    print("=" * 70)
    print(f"Mode:            {args.mode}")
    print(f"Model:           {MODEL_PATH} (ZERO-SHOT, no LoRA)")
    print(f"Top-K cands:     {top_k}")
    print(f"Max code lines:  {max_code_lines}")
    print(f"Max examples:    {args.max_examples}")
    print(f"Device:          {device}")
    print(f"Output dir:      {output_dir}")
    print()
    sys.stdout.flush()

    # --- Load data ---
    data = load_data(args.max_examples, top_k=top_k)
    print(f"Loaded {len(data)} examples")
    sys.stdout.flush()

    # --- Load model (4-bit) ---
    print("Loading model in 4-bit quantization...")
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

    # --- Evaluate ---
    cumulative_hits = {f"hit@{k}": 0 for k in [1, 3, 5, 10, 20]}
    total = 0
    parse_failures = 0
    code_miss_count = 0  # files where code couldn't be read
    prompt_token_stats = []
    per_example_results = []

    t_start = time.time()

    for i, example in enumerate(data):
        repo = example["repo"]
        issue_text = example["issue_text"]
        candidates = example["candidates"]
        gt_files = example["gt_files"]
        num_cands = len(candidates)

        # Read code content for each candidate
        code_contents = []
        for cand_path in candidates:
            code = read_file_lines(repo, cand_path, max_code_lines)
            if code is None:
                code_miss_count += 1
            code_contents.append(code)

        # Build prompt based on mode
        prompt = build_prompt(issue_text, candidates, code_contents)

        # Truncate if needed
        prompt = truncate_prompt(prompt, tokenizer)

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]
        prompt_token_stats.append(prompt_len)

        # Generate
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  [OOM at example {i}] prompt_len={prompt_len}, skipping")
                    sys.stdout.flush()
                    # Count as fallback
                    ranking = list(range(num_cands))
                    is_fallback = True
                    parse_failures += 1
                    ranked_candidates = [candidates[idx] for idx in ranking]
                    hits = compute_metrics(ranked_candidates, gt_files)
                    for k, v in hits.items():
                        cumulative_hits[k] += v
                    total += 1
                    continue
                raise

        # Decode
        gen_tokens = outputs[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Parse ranking
        ranking, is_fallback = parse_ranking(gen_text, num_cands)
        if is_fallback:
            parse_failures += 1

        # Debug: print first 5 examples
        if i < 5:
            print(f"  [DEBUG {i}] prompt_tokens={prompt_len}")
            print(f"  [DEBUG {i}] gen_text (first 300 chars): {gen_text[:300]!r}")
            print(f"  [DEBUG {i}] parsed ranking[:5]: {ranking[:5]} fallback={is_fallback}")
            sys.stdout.flush()

        # Compute metrics
        ranked_candidates = [candidates[idx] for idx in ranking]
        hits = compute_metrics(ranked_candidates, gt_files)
        for k, v in hits.items():
            cumulative_hits[k] += v
        total += 1

        # Store per-example result
        per_example_results.append({
            "repo": repo,
            "issue_id": example["issue_id"],
            "num_candidates": num_cands,
            "prompt_tokens": prompt_len,
            "is_fallback": is_fallback,
            **hits,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = elapsed / total
            eta = rate * (len(data) - total)
            r1 = cumulative_hits["hit@1"] / total * 100
            r5 = cumulative_hits["hit@5"] / total * 100
            r10 = cumulative_hits["hit@10"] / total * 100
            print(f"[{i+1}/{len(data)}] hit@1={r1:.2f}% hit@5={r5:.2f}% hit@10={r10:.2f}% "
                  f"parse_fail={parse_failures}/{total} "
                  f"avg_prompt_tok={np.mean(prompt_token_stats):.0f} "
                  f"ETA={eta/60:.1f}min")
            sys.stdout.flush()

    # --- Final results ---
    elapsed_total = time.time() - t_start
    print()
    print("=" * 70)
    print(f"LISTWISE RERANKER — mode={args.mode}")
    print(f"Model: Qwen2.5-7B-Instruct (ZERO-SHOT, no LoRA, 4-bit)")
    print(f"Top-{top_k} candidates, {total} test examples")
    print(f"Code lines per file: {max_code_lines}")
    print(f"Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)")
    print(f"Code read misses: {code_miss_count}")
    print(f"Prompt tokens — mean: {np.mean(prompt_token_stats):.0f}, "
          f"median: {np.median(prompt_token_stats):.0f}, "
          f"max: {np.max(prompt_token_stats):.0f}")
    print(f"Total time: {elapsed_total/60:.1f} min ({elapsed_total/total:.1f}s/example)")
    print()
    for k in [1, 3, 5, 10, 20]:
        key = f"hit@{k}"
        val = cumulative_hits[key] / total * 100
        print(f"  {key:8s} = {val:.2f}%")
    print("=" * 70)

    # --- Save results ---
    results_file = os.path.join(output_dir, f"results_{args.mode}.json")
    with open(results_file, "w") as f:
        json.dump({
            "mode": args.mode,
            "model": MODEL_PATH,
            "top_k": top_k,
            "max_code_lines": max_code_lines,
            "num_examples": total,
            "parse_failures": parse_failures,
            "code_read_misses": code_miss_count,
            "prompt_token_stats": {
                "mean": float(np.mean(prompt_token_stats)),
                "median": float(np.median(prompt_token_stats)),
                "max": int(np.max(prompt_token_stats)),
                "min": int(np.min(prompt_token_stats)),
            },
            "total_time_seconds": elapsed_total,
            **{key: cumulative_hits[key] / total * 100 for key in cumulative_hits},
            "raw_hits": {key: cumulative_hits[key] for key in cumulative_hits},
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save per-example details
    details_file = os.path.join(output_dir, f"details_{args.mode}.jsonl")
    with open(details_file, "w") as f:
        for rec in per_example_results:
            f.write(json.dumps(rec) + "\n")
    print(f"Per-example details saved to {details_file}")


if __name__ == "__main__":
    main()
