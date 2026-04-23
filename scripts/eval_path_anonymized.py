#!/usr/bin/env python3
"""
Path-anonymization control experiment.

Evaluates the trained reranker with anonymized (hashed) file paths.
Tests whether the model relies on path naming conventions vs genuine
structural knowledge.

Three conditions:
1. original: standard paths (baseline, should match 27.01% R@1)
2. hashed: path components hashed per repo (preserves tree depth + .py ext)
3. shuffled: paths randomly reassigned per example (destroys all path info)

Usage:
    python scripts/eval_path_anonymized.py \
        --condition hashed \
        --gpu_id 0 \
        --output_dir experiments/path_anonymized/hashed
"""
import os
import sys
import json
import hashlib
import argparse
import time
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROMPT_TEMPLATE = (
    "Given the bug report, is this file likely to need modification?\n\n"
    "Bug Report: {issue_text}\n\n"
    "File: {candidate_path}\n\n"
    "Answer:"
)


def hash_path(filepath: str, repo_salt: str) -> str:
    """Hash each path component while preserving tree depth and extension.

    Example: 'src/models/transformer.py' -> 'a3f8b2c1/d9e7f4a0/b1c2d3e4f5a6.py'
    """
    parts = filepath.split('/')
    hashed_parts = []
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            # Last part: preserve extension
            name, ext = os.path.splitext(part)
            # Preserve __init__.py as a special case (structural marker)
            if name == "__init__":
                hashed_parts.append(f"__init__{ext}")
            else:
                h = hashlib.sha256(f"{name}_{repo_salt}".encode()).hexdigest()[:12]
                hashed_parts.append(f"{h}{ext}")
        else:
            # Directory: hash
            h = hashlib.sha256(f"{part}_{repo_salt}".encode()).hexdigest()[:8]
            hashed_parts.append(h)
    return '/'.join(hashed_parts)


def shuffle_path(filepath: str, all_candidates: List[str]) -> str:
    """Return a random different path from the candidate pool."""
    # This is handled differently: we shuffle the mapping externally
    return filepath  # placeholder


def build_prompt(issue_text: str, candidate_path: str) -> str:
    return PROMPT_TEMPLATE.format(
        issue_text=issue_text,
        candidate_path=candidate_path,
    )


def get_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_candidates_batched(model, tokenizer, issue_text, candidates,
                              yes_id, no_id, max_seq_length, device,
                              batch_size=16):
    prompts = [build_prompt(issue_text, cand) for cand in candidates]
    all_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        encodings = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                for prompt in batch_prompts:
                    enc = tokenizer([prompt], return_tensors="pt",
                                   truncation=True, max_length=max_seq_length)
                    ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    out = model(input_ids=ids, attention_mask=mask)
                    logits = out.logits[0, -1]
                    score = (logits[yes_id] - logits[no_id]).item()
                    all_scores.append(score)
                continue
            raise

        logits = outputs.logits
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]
        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


def compute_recall_at_k(predicted, gt, k):
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["original", "hashed", "shuffled"],
                        default="hashed")
    parser.add_argument("--model_path", default="/data/shuyang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path",
                        default=os.path.join(BASE_DIR, "experiments/rankft_runB_graph/best"))
    parser.add_argument("--test_data",
                        default=os.path.join(BASE_DIR, "data/grepo_text/grepo_test.jsonl"))
    parser.add_argument("--bm25_candidates",
                        default=os.path.join(BASE_DIR, "data/rankft/grepo_test_bm25_top500.jsonl"))
    parser.add_argument("--merged_candidates",
                        default=os.path.join(BASE_DIR, "data/rankft/grepo_test_merged_candidates.jsonl"))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--score_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=500)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, f"experiments/path_anonymized/{args.condition}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu_id}"

    print(f"=== Path Anonymization Control: {args.condition} ===")

    # Load test data
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item.get("changed_py_files"):
                test_data.append(item)
    print(f"  {len(test_data)} test examples")

    # Load candidates (try merged first, fallback to BM25)
    candidates_path = args.merged_candidates if os.path.exists(args.merged_candidates) else args.bm25_candidates
    print(f"Loading candidates from {candidates_path}...")
    cand_map = {}
    with open(candidates_path) as f:
        for line in f:
            item = json.loads(line)
            key = f"{item['repo']}_{item['issue_id']}"
            cand_map[key] = item.get("candidates", item.get("bm25_candidates", []))
    print(f"  Candidates for {len(cand_map)} examples")

    # Load model
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print("  Model loaded.")

    # Evaluate
    k_values = [1, 3, 5, 10, 20]
    overall = {f"recall@{k}": [] for k in k_values}
    cond_acc1_correct = 0
    cond_acc1_total = 0
    results = []

    total = len(test_data)
    t0 = time.time()
    skipped = 0

    for idx, ex in enumerate(test_data):
        repo = ex["repo"]
        issue_id = ex["issue_id"]
        issue_text = ex["issue_text"]
        gt_files = set(ex["changed_py_files"])
        key = f"{repo}_{issue_id}"

        candidates = cand_map.get(key, [])[:args.top_k]
        if not candidates:
            skipped += 1
            continue

        gt_in_pool = bool(gt_files & set(candidates))

        # Apply path anonymization to candidates
        if args.condition == "original":
            display_candidates = candidates[:]
        elif args.condition == "hashed":
            repo_salt = hashlib.sha256(repo.encode()).hexdigest()[:16]
            display_candidates = [hash_path(c, repo_salt) for c in candidates]
        elif args.condition == "shuffled":
            # Random permutation of paths — destroys all path info
            rng = random.Random(hash((repo, issue_id)))
            display_candidates = candidates[:]
            rng.shuffle(display_candidates)
            # Now score with shuffled paths, but track GT by original position
            # NOTE: this means the model sees wrong paths for each file
            # The GT mapping still uses original paths

        # Score with (possibly anonymized) paths
        scores = score_candidates_batched(
            model, tokenizer, issue_text, display_candidates,
            yes_id, no_id, args.max_seq_length, device, args.score_batch_size,
        )

        # For hashed/original: rerank and map back to original paths
        if args.condition in ("original", "hashed"):
            scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
            reranked = [c for c, _ in scored]
        elif args.condition == "shuffled":
            # Scores correspond to shuffled paths, not real files
            # Map scores back to original candidates by position
            scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
            reranked = [c for c, _ in scored]

        for k in k_values:
            r = compute_recall_at_k(reranked, gt_files, k)
            overall[f"recall@{k}"].append(r)

        if gt_in_pool:
            cond_acc1_total += 1
            if reranked[0] in gt_files:
                cond_acc1_correct += 1

        results.append({
            "repo": repo,
            "issue_id": issue_id,
            "predicted": reranked[:20],
            "ground_truth": list(gt_files),
        })

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            r1 = np.mean(overall["recall@1"]) * 100 if overall["recall@1"] else 0
            r5 = np.mean(overall["recall@5"]) * 100 if overall["recall@5"] else 0
            print(f"  [{idx+1}/{total}] R@1: {r1:.2f}% R@5: {r5:.2f}% ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"Path Anonymization Control: {args.condition}")
    print(f"{'='*60}")
    print(f"Evaluated: {len(overall['recall@1'])} examples in {elapsed:.0f}s (skipped {skipped})")
    print()
    for k in k_values:
        vals = overall[f"recall@{k}"]
        avg = np.mean(vals) * 100 if vals else 0
        print(f"  Recall@{k:<5} {avg:>7.2f}%")
    if cond_acc1_total > 0:
        print(f"  Cond.Acc@1:  {cond_acc1_correct/cond_acc1_total*100:.2f}% ({cond_acc1_correct}/{cond_acc1_total})")

    # Save
    summary = {
        "condition": args.condition,
        "overall": {f"recall@{k}": float(np.mean(overall[f"recall@{k}"]) * 100) for k in k_values},
        "cond_acc1": cond_acc1_correct / max(cond_acc1_total, 1) * 100,
        "n_samples": len(overall["recall@1"]),
        "elapsed": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
