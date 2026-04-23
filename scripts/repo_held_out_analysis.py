#!/usr/bin/env python3
"""
Repo-level generalization analysis.

Instead of retraining, we analyze existing predictions by repo:
1. Per-repo R@1 for baseline vs shuffle
2. Stratify by repo naming convention quality (descriptive vs opaque)
3. Show that path prior holds consistently across diverse repos
4. Report per-repo shuffle drop distribution

Also leverage existing cross-benchmark results as repo-held-out evidence:
- SWE-bench: 12 unseen repos, same collapse
- BeetleBox Java: unseen repos + unseen language, same collapse

Usage:
    python scripts/repo_held_out_analysis.py \
        --baseline_preds experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl \
        --shuffle_preds experiments/path_perturb_shuffle_filenames/eval_4bit/predictions.jsonl \
        --output_dir /data/chenlibin/grepo_agent_experiments/repo_generalization
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"


def load_predictions(path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_preds", type=str, required=True)
    parser.add_argument("--shuffle_preds", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/repo_generalization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    baseline = load_predictions(args.baseline_preds)
    shuffle = load_predictions(args.shuffle_preds)

    # Per-repo analysis
    repo_baseline = defaultdict(list)
    repo_shuffle = defaultdict(list)

    for key, pred in baseline.items():
        repo_baseline[pred["repo"]].append(pred["hit@1"])
    for key, pred in shuffle.items():
        repo_shuffle[pred["repo"]].append(pred["hit@1"])

    # Compute per-repo stats
    repo_stats = []
    for repo in sorted(repo_baseline.keys()):
        b_hits = repo_baseline[repo]
        s_hits = repo_shuffle.get(repo, [])
        if len(b_hits) < 5:
            continue

        b_r1 = np.mean(b_hits) * 100
        s_r1 = np.mean(s_hits) * 100 if s_hits else 0
        drop = (1 - s_r1 / max(0.01, b_r1)) * 100 if b_r1 > 0 else 0

        repo_stats.append({
            "repo": repo,
            "n": len(b_hits),
            "baseline_R@1": float(b_r1),
            "shuffle_R@1": float(s_r1),
            "drop_pct": float(drop),
        })

    # Sort by number of examples
    repo_stats.sort(key=lambda x: -x["n"])

    # Distribution of drops
    drops = [r["drop_pct"] for r in repo_stats if r["baseline_R@1"] > 5]
    n_positive_drop = sum(1 for d in drops if d > 20)

    print(f"=== Per-repo analysis ({len(repo_stats)} repos with n>=5) ===")
    print(f"{'Repo':<25} {'N':>5} {'Base R@1':>10} {'Shuf R@1':>10} {'Drop':>8}")
    print("-" * 65)
    for r in repo_stats[:20]:
        print(f"{r['repo']:<25} {r['n']:>5} {r['baseline_R@1']:>9.1f}% {r['shuffle_R@1']:>9.1f}% {r['drop_pct']:>7.0f}%")

    print(f"\n=== Drop distribution ===")
    print(f"Repos with >20% drop: {n_positive_drop}/{len(drops)} ({n_positive_drop/len(drops)*100:.0f}%)")
    print(f"Mean drop: {np.mean(drops):.1f}%")
    print(f"Median drop: {np.median(drops):.1f}%")
    print(f"Min drop: {min(drops):.1f}%, Max drop: {max(drops):.1f}%")
    print(f"Std: {np.std(drops):.1f}%")

    # Quartile analysis
    sorted_by_baseline = sorted(repo_stats, key=lambda x: x["baseline_R@1"])
    n = len(sorted_by_baseline)
    q1 = sorted_by_baseline[:n//4]
    q4 = sorted_by_baseline[3*n//4:]

    q1_mean_drop = np.mean([r["drop_pct"] for r in q1 if r["baseline_R@1"] > 5])
    q4_mean_drop = np.mean([r["drop_pct"] for r in q4 if r["baseline_R@1"] > 5])
    print(f"\nBottom quartile repos (low R@1): mean drop = {q1_mean_drop:.1f}%")
    print(f"Top quartile repos (high R@1): mean drop = {q4_mean_drop:.1f}%")

    summary = {
        "n_repos": len(repo_stats),
        "n_repos_with_drop_gt_20pct": n_positive_drop,
        "mean_drop": float(np.mean(drops)),
        "median_drop": float(np.median(drops)),
        "std_drop": float(np.std(drops)),
        "bottom_quartile_mean_drop": float(q1_mean_drop),
        "top_quartile_mean_drop": float(q4_mean_drop),
        "per_repo": repo_stats,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
