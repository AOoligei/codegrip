#!/usr/bin/env python3
"""
Repo-level analysis: does path-only vs code-aware gap hold across all repos?

For each repo, compute:
1. Path-only R@1
2. Code-aware R@1 (from existing predictions)
3. Shuffle R@1
4. Path-code gap

This addresses reviewer Q4 without requiring re-training.

Usage:
    python scripts/repo_holdout_analysis.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/repo_analysis
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

np.random.seed(42)


def load_predictions(path):
    """Load predictions.jsonl, return dict keyed by (repo, issue_id)."""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec.get("issue_id", "")))
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            hit1 = len(set(predicted[:1]) & gt) / max(1, len(gt))
            preds[key] = {"hit@1": hit1, "repo": rec["repo"]}
    return preds


def repo_level_comparison(path_preds, shuffle_preds=None, code_preds=None):
    """Compute per-repo R@1 for path-only, shuffle, and code-aware."""
    repo_results = defaultdict(lambda: {
        "path_hits": [], "shuffle_hits": [], "code_hits": []
    })

    for key, p in path_preds.items():
        repo = p["repo"]
        repo_results[repo]["path_hits"].append(p["hit@1"])
        if shuffle_preds and key in shuffle_preds:
            repo_results[repo]["shuffle_hits"].append(shuffle_preds[key]["hit@1"])
        if code_preds and key in code_preds:
            repo_results[repo]["code_hits"].append(code_preds[key]["hit@1"])

    summary = []
    for repo, data in repo_results.items():
        n = len(data["path_hits"])
        if n < 5:
            continue
        entry = {
            "repo": repo,
            "n_examples": n,
            "path_R@1": float(np.mean(data["path_hits"]) * 100),
        }
        if data["shuffle_hits"]:
            entry["shuffle_R@1"] = float(np.mean(data["shuffle_hits"]) * 100)
            entry["shuffle_drop"] = entry["path_R@1"] - entry["shuffle_R@1"]
        if data["code_hits"]:
            entry["code_R@1"] = float(np.mean(data["code_hits"]) * 100)
            entry["code_gap"] = entry["code_R@1"] - entry["path_R@1"]
        summary.append(entry)

    return sorted(summary, key=lambda x: -x["n_examples"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_preds", type=str,
                        default="experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl")
    parser.add_argument("--shuffle_preds", type=str,
                        default="experiments/path_perturb_shuffle_filenames/eval_4bit/predictions.jsonl")
    parser.add_argument("--code_preds", type=str, default=None,
                        help="Optional: code-aware model predictions")
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/repo_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions...")
    path_preds = load_predictions(args.path_preds)
    print(f"  Path-only: {len(path_preds)} examples")

    shuffle_preds = None
    if os.path.isfile(args.shuffle_preds):
        shuffle_preds = load_predictions(args.shuffle_preds)
        print(f"  Shuffle: {len(shuffle_preds)} examples")

    code_preds = None
    if args.code_preds and os.path.isfile(args.code_preds):
        code_preds = load_predictions(args.code_preds)
        print(f"  Code-aware: {len(code_preds)} examples")

    print("\nPer-repo analysis...")
    results = repo_level_comparison(path_preds, shuffle_preds, code_preds)

    # Summary stats
    path_r1s = [r["path_R@1"] for r in results]
    print(f"\n  {len(results)} repos with >=5 examples")
    print(f"  Path R@1: mean={np.mean(path_r1s):.1f}%, "
          f"min={np.min(path_r1s):.1f}%, max={np.max(path_r1s):.1f}%")

    if any("shuffle_R@1" in r for r in results):
        drops = [r["shuffle_drop"] for r in results if "shuffle_drop" in r]
        n_positive_drop = sum(1 for d in drops if d > 0)
        print(f"  Shuffle drop: mean={np.mean(drops):.1f}pp, "
              f"positive in {n_positive_drop}/{len(drops)} repos")

    if any("code_gap" in r for r in results):
        gaps = [r["code_gap"] for r in results if "code_gap" in r]
        n_code_better = sum(1 for g in gaps if g > 0)
        print(f"  Code gap: mean={np.mean(gaps):.1f}pp, "
              f"code better in {n_code_better}/{len(gaps)} repos")

    # Top/bottom repos
    print("\n  Top 5 repos by path R@1:")
    for r in sorted(results, key=lambda x: -x["path_R@1"])[:5]:
        line = f"    {r['repo']}: path={r['path_R@1']:.1f}%"
        if "shuffle_R@1" in r:
            line += f" shuffle={r['shuffle_R@1']:.1f}%"
        print(line)

    print("\n  Bottom 5 repos by path R@1:")
    for r in sorted(results, key=lambda x: x["path_R@1"])[:5]:
        line = f"    {r['repo']}: path={r['path_R@1']:.1f}%"
        if "shuffle_R@1" in r:
            line += f" shuffle={r['shuffle_R@1']:.1f}%"
        print(line)

    with open(os.path.join(args.output_dir, "repo_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output_dir}/repo_analysis.json")


if __name__ == "__main__":
    main()
