#!/usr/bin/env python3
"""
Code-Crucial sensitivity analysis: how stable are the Code-Crucial subset
conclusions under different construction thresholds?

Varies the path-overlap threshold and checks whether the conclusion
"code-residual underperforms path-only on hard examples" holds.

Usage:
    python scripts/codecrucial_sensitivity.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/codecrucial_sensitivity
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"

# Prediction files for different models
PRED_PATHS = {
    "path_only": "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
}


def compute_overlap(issue_text, file_paths):
    """Compute token overlap between issue and file paths."""
    import re
    issue_tokens = set(re.split(r'\W+', issue_text.lower()))
    path_tokens = set()
    for p in file_paths:
        path_tokens.update(re.split(r'[/._\-]', p.lower()))
    path_tokens.discard('')
    issue_tokens.discard('')
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def load_predictions(path):
    """Load predictions keyed by (repo, issue_id)."""
    preds = {}
    if not os.path.isfile(path):
        return preds
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            hit1 = len(set(predicted[:1]) & gt) / max(1, len(gt))
            preds[key] = {"hit@1": hit1, "gt": gt, "predicted": predicted}
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/codecrucial_sensitivity")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    # Load predictions
    model_preds = {}
    for name, path in PRED_PATHS.items():
        full_path = os.path.join("/home/chenlibin/grepo_agent", path)
        model_preds[name] = load_predictions(full_path)

    # Compute overlap for each example
    overlaps = {}
    for key, rec in test_data.items():
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        overlap = compute_overlap(rec["issue_text"], gt_files)
        overlaps[key] = overlap

    # Sweep thresholds
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    print(f"{'Threshold':>10} {'N':>6} {'Path-only R@1':>15}")
    print("-" * 35)

    results = []
    for thresh in thresholds:
        # Select examples where overlap <= threshold (hard examples)
        hard_keys = [k for k, o in overlaps.items() if o <= thresh]

        if not hard_keys:
            continue

        # Compute R@1 for each model on this subset
        row = {"threshold": thresh, "n_examples": len(hard_keys)}
        for name, preds in model_preds.items():
            hits = [preds[k]["hit@1"] for k in hard_keys if k in preds]
            if hits:
                r1 = np.mean(hits) * 100
                row[f"{name}_R@1"] = r1
            else:
                row[f"{name}_R@1"] = None

        results.append(row)
        path_r1 = row.get("path_only_R@1", 0) or 0
        print(f"{thresh:>10.2f} {len(hard_keys):>6} {path_r1:>15.2f}%")

    # Also compute by percentile buckets
    print(f"\n{'Percentile':>10} {'N':>6} {'Overlap range':>15} {'Path-only R@1':>15}")
    print("-" * 50)

    overlap_values = list(overlaps.values())
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    bucket_results = []
    for i in range(len(percentiles) - 1):
        lo = np.percentile(overlap_values, percentiles[i])
        hi = np.percentile(overlap_values, percentiles[i + 1])
        bucket_keys = [k for k, o in overlaps.items() if lo <= o <= hi]
        if not bucket_keys:
            continue

        hits = [model_preds["path_only"][k]["hit@1"]
                for k in bucket_keys if k in model_preds["path_only"]]
        r1 = np.mean(hits) * 100 if hits else 0
        bucket_results.append({
            "percentile": f"{percentiles[i]}-{percentiles[i+1]}",
            "n": len(bucket_keys),
            "overlap_lo": float(lo),
            "overlap_hi": float(hi),
            "path_only_R@1": r1,
        })
        print(f"{percentiles[i]:>3}-{percentiles[i+1]:>3}%  {len(bucket_keys):>6} "
              f"[{lo:.3f}, {hi:.3f}] {r1:>15.2f}%")

    summary = {
        "threshold_sweep": results,
        "percentile_buckets": bucket_results,
        "total_examples": len(overlaps),
        "median_overlap": float(np.median(overlap_values)),
        "mean_overlap": float(np.mean(overlap_values)),
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMedian overlap: {summary['median_overlap']:.3f}")
    print(f"Mean overlap: {summary['mean_overlap']:.3f}")
    print(f"Saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
