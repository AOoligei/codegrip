#!/usr/bin/env python3
"""
Downstream validity: gold-file conditioned function-level success.

For each example where the file-level top-1 prediction is correct:
  - Does the correct file contain the GT function?
  - How many functions does the file have? (difficulty proxy)
  - What fraction of file-level successes also succeed at function-level?

This directly addresses reviewer Q2: "is file-level R@1 a valid proxy
for code understanding?"

Usage:
    python scripts/file_to_function_gap.py \
        --predictions experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl \
        --output_dir /data/chenlibin/grepo_agent_experiments/file_to_function_gap
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
FUNC_INDEX_PATH = "/home/chenlibin/grepo_agent/data/function_index_aligned.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/file_to_function_gap")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.predictions]

    # Load test data
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    func_index = json.load(open(FUNC_INDEX_PATH))

    results = []

    for pred_path, label in zip(args.predictions, labels):
        if not os.path.isfile(pred_path):
            print(f"  SKIP {label}: {pred_path} not found")
            continue

        # Load predictions
        preds = {}
        with open(pred_path) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["repo"], str(rec.get("issue_id", "")))
                preds[key] = rec

        # Analyze
        file_hits = 0
        file_misses = 0
        file_hit_func_hit = 0
        file_hit_func_miss = 0
        file_hit_no_gt_func = 0
        func_counts = []  # number of functions in correctly predicted files
        per_repo = defaultdict(lambda: {"file_hit": 0, "func_hit": 0, "total": 0})

        for key, pred_rec in preds.items():
            if key not in test_data:
                continue

            test_rec = test_data[key]
            gt_files = set(test_rec.get("changed_py_files",
                                         test_rec.get("changed_files", [])))
            gt_funcs = set(test_rec.get("changed_functions", []))
            predicted = pred_rec.get("predicted", [])
            repo = test_rec["repo"]

            per_repo[repo]["total"] += 1

            # File-level check
            top1_file = predicted[0] if predicted else ""
            if top1_file not in gt_files:
                file_misses += 1
                continue

            file_hits += 1
            per_repo[repo]["file_hit"] += 1

            if not gt_funcs:
                file_hit_no_gt_func += 1
                continue

            # Function-level check: does top1_file contain any GT function?
            repo_idx = func_index.get(repo, {})
            file_funcs = set(repo_idx.get(top1_file, []))
            func_counts.append(len(file_funcs))

            if gt_funcs & file_funcs:
                file_hit_func_hit += 1
                per_repo[repo]["func_hit"] += 1
            else:
                file_hit_func_miss += 1

        # Per-repo correlation
        repo_file_r1 = []
        repo_func_rate = []
        for repo, stats in per_repo.items():
            if stats["total"] >= 5 and stats["file_hit"] >= 2:
                repo_file_r1.append(stats["file_hit"] / stats["total"])
                repo_func_rate.append(stats["func_hit"] / max(1, stats["file_hit"]))

        correlation = float(np.corrcoef(repo_file_r1, repo_func_rate)[0, 1]) if len(repo_file_r1) >= 5 else 0.0

        total_with_func = file_hit_func_hit + file_hit_func_miss
        func_rate = file_hit_func_hit / max(1, total_with_func)

        result = {
            "label": label,
            "file_hits": file_hits,
            "file_misses": file_misses,
            "file_hit_func_hit": file_hit_func_hit,
            "file_hit_func_miss": file_hit_func_miss,
            "file_hit_no_gt_func": file_hit_no_gt_func,
            "file_to_func_rate": float(func_rate),
            "mean_funcs_in_hit_file": float(np.mean(func_counts)) if func_counts else 0,
            "repo_correlation": correlation,
            "n_repos_for_corr": len(repo_file_r1),
        }
        results.append(result)

        print(f"\n{label}:")
        print(f"  File hits: {file_hits} / {file_hits + file_misses}")
        print(f"  Of file hits with GT funcs:")
        print(f"    Function also hit: {file_hit_func_hit} / {total_with_func} ({func_rate:.1%})")
        print(f"    Function missed:   {file_hit_func_miss} / {total_with_func}")
        print(f"    No GT func label:  {file_hit_no_gt_func}")
        print(f"  Mean funcs in hit file: {result['mean_funcs_in_hit_file']:.1f}")
        print(f"  Repo-level corr(file_R@1, func_rate): {correlation:.3f} (n={len(repo_file_r1)} repos)")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
