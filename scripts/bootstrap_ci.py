#!/usr/bin/env python3
"""
Compute repo-level bootstrap confidence intervals for key results.

Reads predictions.jsonl from eval outputs, computes R@1 with 95% CI
via repo-level bootstrap (resample repos, not examples).

Usage:
    python scripts/bootstrap_ci.py \
        --predictions experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl \
        --label "Path-only (Qwen2.5-7B)" \
        --n_bootstrap 10000
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

np.random.seed(42)


def load_predictions(path):
    """Load predictions.jsonl, return list of dicts with repo + hit@1."""
    preds = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            hit1 = len(set(predicted[:1]) & gt) / max(1, len(gt))
            preds.append({
                "repo": rec.get("repo", "unknown"),
                "hit@1": hit1,
            })
    return preds


def repo_bootstrap_ci(preds, n_bootstrap=10000, ci=0.95):
    """Compute R@1 with repo-level bootstrap CI."""
    # Group by repo
    repo_hits = defaultdict(list)
    for p in preds:
        repo_hits[p["repo"]].append(p["hit@1"])

    repos = list(repo_hits.keys())
    n_repos = len(repos)

    # Overall mean
    overall = np.mean([p["hit@1"] for p in preds])

    # Bootstrap: resample repos with replacement
    boot_means = []
    for _ in range(n_bootstrap):
        sampled_repos = np.random.choice(repos, size=n_repos, replace=True)
        sampled_hits = []
        for r in sampled_repos:
            sampled_hits.extend(repo_hits[r])
        boot_means.append(np.mean(sampled_hits))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)

    return {
        "mean": float(overall * 100),
        "ci_lo": float(lo * 100),
        "ci_hi": float(hi * 100),
        "n_examples": len(preds),
        "n_repos": n_repos,
        "n_bootstrap": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, nargs="+", required=True,
                        help="Path(s) to predictions.jsonl files")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.predictions]

    results = []
    for path, label in zip(args.predictions, labels):
        if not os.path.isfile(path):
            print(f"  SKIP {label}: {path} not found")
            continue

        preds = load_predictions(path)
        ci = repo_bootstrap_ci(preds, args.n_bootstrap)
        ci["label"] = label
        results.append(ci)
        print(f"  {label}: R@1 = {ci['mean']:.2f}% "
              f"[{ci['ci_lo']:.2f}, {ci['ci_hi']:.2f}] "
              f"({ci['n_examples']} ex, {ci['n_repos']} repos)")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
