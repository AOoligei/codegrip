#!/usr/bin/env python3
"""
Ensemble multiple SWE-bench adapted reranker checkpoints.

Strategy 1: Rank fusion (RRF - Reciprocal Rank Fusion)
Strategy 2: Score averaging (if scores available)
Strategy 3: Majority vote on top-1

Input: multiple predictions.jsonl files
Output: ensembled R@1

Usage:
    python scripts/ensemble_swebench.py \
        --predictions path1 path2 path3 ... \
        --output_dir /data/chenlibin/grepo_agent_experiments/swebench_ensemble
"""

import argparse
import json
import os
from collections import defaultdict


def load_predictions(path):
    """Load predictions.jsonl, return dict of (repo, issue_id) -> record."""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec.get("repo", ""), str(rec.get("issue_id", "")))
            preds[key] = rec
    return preds


def rrf_fusion(ranked_lists, k=60):
    """Reciprocal Rank Fusion.

    ranked_lists: list of ranked candidate lists
    Returns: fused ranking (candidate -> RRF score)
    """
    scores = defaultdict(float)
    for ranking in ranked_lists:
        for rank, cand in enumerate(ranking):
            scores[cand] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def compute_hit_at_k(predicted, gt_files, k):
    """Hit@k: is any GT file in top-k."""
    if not gt_files:
        return 0.0
    top_k = set(predicted[:k])
    return 1.0 if (top_k & set(gt_files)) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, nargs="+", required=True,
                        help="Paths to predictions.jsonl files")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Optional labels for each predictions file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rrf_k", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels or [os.path.basename(os.path.dirname(p))
                              for p in args.predictions]

    print("Loading predictions...")
    all_preds = []
    for path, label in zip(args.predictions, labels):
        preds = load_predictions(path)
        print(f"  {label}: {len(preds)} examples")
        all_preds.append((label, preds))

    # Find common keys
    common_keys = set(all_preds[0][1].keys())
    for _, p in all_preds[1:]:
        common_keys &= set(p.keys())
    print(f"\n  Common examples: {len(common_keys)}")

    # Compute individual h@1
    print("\n=== Individual R@1 ===")
    individual_hits = {}
    for label, preds in all_preds:
        hits = []
        for key in common_keys:
            rec = preds[key]
            predicted = rec.get("predicted", [])
            gt = rec.get("ground_truth", [])
            hits.append(compute_hit_at_k(predicted, gt, 1))
        r1 = sum(hits) / len(hits) * 100
        individual_hits[label] = hits
        print(f"  {label}: h@1 = {r1:.2f}%")

    # Compute agreement
    print("\n=== Agreement analysis ===")
    all_correct = 0
    any_correct = 0
    for key in common_keys:
        hits = [individual_hits[l][list(common_keys).index(key)] for l in individual_hits]
        if all(h > 0 for h in hits):
            all_correct += 1
        if any(h > 0 for h in hits):
            any_correct += 1
    print(f"  All models correct: {all_correct / len(common_keys) * 100:.2f}%")
    print(f"  Any model correct:  {any_correct / len(common_keys) * 100:.2f}% (oracle upper bound)")

    # Ensemble via RRF
    print("\n=== RRF Ensemble ===")
    ensemble_hits = []
    for key in common_keys:
        ranked_lists = []
        gt_files = None
        for _, preds in all_preds:
            rec = preds[key]
            predicted = rec.get("predicted", [])
            ranked_lists.append(predicted)
            if gt_files is None:
                gt_files = rec.get("ground_truth", [])

        fused = rrf_fusion(ranked_lists, k=args.rrf_k)
        fused_cands = [c for c, _ in fused]
        ensemble_hits.append(compute_hit_at_k(fused_cands, gt_files, 1))

    ensemble_r1 = sum(ensemble_hits) / len(ensemble_hits) * 100
    print(f"  RRF h@1 = {ensemble_r1:.2f}%")

    # Save
    summary = {
        "num_examples": len(common_keys),
        "individual": {l: sum(h) / len(h) * 100 for l, h in individual_hits.items()},
        "ensemble_rrf_h@1": ensemble_r1,
        "all_correct_pct": all_correct / len(common_keys) * 100,
        "any_correct_pct": any_correct / len(common_keys) * 100,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
