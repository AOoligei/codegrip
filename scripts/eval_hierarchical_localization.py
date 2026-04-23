#!/usr/bin/env python3
"""
Hierarchical Localization Experiment.

Tests: "Code semantics matter only AFTER the parent scope is identified."

Three modes:
  global       - rank ALL functions across ALL files (~1200 candidates)
  gold_file    - rank functions WITHIN ground-truth file(s) only (~10-30 candidates)
  gold_dir     - rank functions within GT file's directory (~50-200 candidates)

If path-shuffle collapse is large in global but small in gold_file,
it proves global localization = naming, conditioned localization = semantics.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from eval_function_level import (
    resolve_gt_function_pairs,
    load_model,
    PROMPT_TEMPLATE,
)


def get_yes_no_ids(tokenizer):
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    return yes_ids[0], no_ids[0]


@torch.no_grad()
def score_functions(model, tokenizer, issue_text, candidates, yes_id, no_id,
                    device, max_seq_length=768, batch_size=16):
    """Score function candidates. candidates = list of (file_path, func_name)."""
    if not candidates:
        return []

    prompts = [
        PROMPT_TEMPLATE.format(
            issue_text=issue_text,
            file_path=fpath,
            function_name=fname,
        )
        for fpath, fname in candidates
    ]

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_seq_length).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        seq_lengths = inputs.attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, seq_lengths]
        scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
        all_scores.extend(scores)

    return all_scores


def build_shuffle_display_map(candidates, rng):
    """Create a mapping from real (fpath, fname) to shuffled display names.
    Keeps file paths fixed, shuffles function names. Returns dict mapping
    (fpath, fname) -> display_fname for prompt construction.
    Real identities are preserved for metric computation."""
    if len(candidates) <= 1:
        return {c: c[1] for c in candidates}
    fnames = [c[1] for c in candidates]
    shuffled = list(fnames)
    rng.shuffle(shuffled)
    return {c: shuffled[i] for i, c in enumerate(candidates)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--mode", choices=["global", "gold_file", "gold_dir"], required=True)
    parser.add_argument("--perturb", choices=["none", "shuffle"], default="none")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Load data
    test_data = [json.loads(l) for l in open(BASE_DIR / "data/grepo_text/grepo_test.jsonl")]
    func_index = json.load(open(BASE_DIR / "data/function_index_aligned.json"))

    # BM25 candidates (only for global mode)
    bm25_cands = {}
    if args.mode == "global":
        for l in open(BASE_DIR / "data/rankft/merged_bm25_exp6_candidates.jsonl"):
            d = json.loads(l)
            bm25_cands[(d["repo"], d["issue_id"])] = d["candidates"]

    # Load model
    model, tokenizer, yes_id, no_id, device = load_model(
        args.model_path, args.lora_path, args.gpu_id)

    print(f"Mode: {args.mode}, Perturb: {args.perturb}", flush=True)

    rng = random.Random(42)
    results = {"R@1": [], "R@5": [], "Hit@1": [], "n_candidates": []}
    n_evaluated = 0
    n_skipped = 0
    t0 = time.time()

    for idx, ex in enumerate(test_data):
        repo = ex["repo"]
        changed_funcs = ex.get("changed_functions", [])
        changed_files = ex.get("changed_py_files", [])
        issue_text = ex["issue_text"]

        if not changed_funcs or not changed_files:
            n_skipped += 1
            continue

        repo_funcs = func_index.get(repo, {})
        gt_pairs, _ = resolve_gt_function_pairs(changed_funcs, changed_files, repo_funcs)
        if not gt_pairs:
            n_skipped += 1
            continue

        # Build candidate set based on mode
        candidates = []  # list of (file_path, func_name)

        if args.mode == "global":
            # Same as eval_function_level: expand top-50 BM25 files
            key = (repo, ex["issue_id"])
            if key not in bm25_cands:
                n_skipped += 1
                continue
            top_files = bm25_cands[key][:50]
            for fpath in top_files:
                for fname in repo_funcs.get(fpath, []):
                    candidates.append((fpath, fname))

        elif args.mode == "gold_file":
            # Only functions in GT file(s)
            for fpath in changed_files:
                for fname in repo_funcs.get(fpath, []):
                    candidates.append((fpath, fname))

        elif args.mode == "gold_dir":
            # All functions in GT file's directory
            gt_dirs = set(os.path.dirname(f) for f in changed_files)
            for fpath, funcs in repo_funcs.items():
                if os.path.dirname(fpath) in gt_dirs:
                    for fname in funcs:
                        candidates.append((fpath, fname))

        if not candidates:
            n_skipped += 1
            continue

        # Build display map (shuffled or identity)
        if args.perturb == "shuffle":
            display_map = build_shuffle_display_map(candidates, rng)
        else:
            display_map = {c: c[1] for c in candidates}

        # Score using display names in prompts
        prompts = [
            PROMPT_TEMPLATE.format(
                issue_text=issue_text,
                file_path=c[0],
                function_name=display_map[c],
            )
            for c in candidates
        ]

        all_scores = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=768).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            seq_lengths = inputs.attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=device)
            last_logits = logits[batch_indices, seq_lengths]
            scores = (last_logits[:, yes_id] - last_logits[:, no_id]).cpu().tolist()
            all_scores.extend(scores)

        # Rank using REAL identities (not display names)
        ranked_indices = np.argsort([-s for s in all_scores])
        ranked = [f"{candidates[i][0]}::{candidates[i][1]}" for i in ranked_indices]

        # Metrics
        n_gt = len(gt_pairs)
        for k in [1, 5]:
            top_k = set(ranked[:k])
            results[f"R@{k}"].append(len(top_k & gt_pairs) / n_gt * 100)
        results["Hit@1"].append(100.0 if set(ranked[:1]) & gt_pairs else 0.0)
        results["n_candidates"].append(len(candidates))
        n_evaluated += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            r1 = np.mean(results["R@1"])
            print(f"  [{idx+1}/{len(test_data)}] R@1={r1:.2f}% "
                  f"avg_cands={np.mean(results['n_candidates']):.0f} "
                  f"({elapsed:.0f}s)", flush=True)

    # Summary
    summary = {
        "mode": args.mode,
        "perturb": args.perturb,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "avg_candidates": float(np.mean(results["n_candidates"])),
        "median_candidates": float(np.median(results["n_candidates"])),
        "R@1": float(np.mean(results["R@1"])),
        "R@5": float(np.mean(results["R@5"])),
        "Hit@1": float(np.mean(results["Hit@1"])),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.mode} | perturb={args.perturb} ===")
    print(f"  Evaluated: {n_evaluated}, Skipped: {n_skipped}")
    print(f"  Avg candidates: {summary['avg_candidates']:.0f}")
    print(f"  R@1={summary['R@1']:.2f}%  R@5={summary['R@5']:.2f}%  Hit@1={summary['Hit@1']:.2f}%")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
