#!/usr/bin/env python3
"""
Collect multi-seed experiment results and compute mean +/- std for paper reporting.

Usage:
    python scripts/collect_multiseed_results.py \
        --results_dirs path/to/seed42/summary.json path/to/seed123/summary.json ...

    # Or auto-discover from base directory:
    python scripts/collect_multiseed_results.py \
        --base_dir /data/chenlibin/grepo_agent_experiments \
        --seeds 42 123 456
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20"]
# Aliases used in summary.json
METRIC_ALIASES = {
    "recall@1": "hit@1",
    "recall@3": "hit@3",
    "recall@5": "hit@5",
    "recall@10": "hit@10",
    "recall@20": "hit@20",
}


def find_summary(seed_dir: str) -> str | None:
    """Find summary.json inside a seed experiment directory, trying common eval subdirs."""
    # Only use eval_graph_rerank — the standardized eval with merged_bm25_exp6_candidates.jsonl
    candidates = [
        os.path.join(seed_dir, "eval_graph_rerank", "summary.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Collect multi-seed results")
    parser.add_argument("--results_dirs", nargs="+", default=None,
                        help="Explicit paths to summary.json files")
    parser.add_argument("--base_dir", default="/data/chenlibin/grepo_agent_experiments",
                        help="Base directory containing multiseed_seed* dirs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Seed values to collect")
    args = parser.parse_args()

    # Resolve summary paths
    summary_paths = []
    if args.results_dirs:
        summary_paths = args.results_dirs
    else:
        for seed in args.seeds:
            seed_dir = os.path.join(args.base_dir, f"multiseed_seed{seed}")
            found = find_summary(seed_dir)
            if found:
                summary_paths.append(found)
            else:
                print(f"WARNING: No summary.json found for seed {seed} in {seed_dir}")

    if len(summary_paths) < 2:
        print(f"ERROR: Need at least 2 seed results, found {len(summary_paths)}")
        sys.exit(1)

    # Load all summaries
    summaries = []
    for p in summary_paths:
        if not os.path.isfile(p):
            print(f"WARNING: {p} not found, skipping")
            continue
        data = load_summary(p)
        summaries.append(data)
        print(f"Loaded: {p}")

    n_seeds = len(summaries)
    print(f"\n{'='*60}")
    print(f"Multi-seed results ({n_seeds} seeds)")
    print(f"{'='*60}\n")

    # --- Overall metrics ---
    print("Overall metrics (mean +/- std):")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Seeds':>30}")
    print("-" * 62)

    metric_results = {}
    for metric in METRICS:
        values = []
        for s in summaries:
            overall = s.get("overall", {})
            val = overall.get(metric)
            if val is None:
                # Try alias (recall@k -> hit@k)
                for alias, target in METRIC_ALIASES.items():
                    if target == metric and alias in overall:
                        val = overall[alias]
                        break
            if val is not None:
                values.append(val)

        if len(values) == n_seeds:
            arr = np.array(values)
            mean = arr.mean()
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            seeds_str = ", ".join(f"{v:.2f}" for v in values)
            print(f"{metric:<12} {mean:>8.2f} {std:>8.2f}   [{seeds_str}]")
            metric_results[metric] = {"mean": float(mean), "std": float(std),
                                      "values": [float(v) for v in values]}
        else:
            print(f"{metric:<12} {'N/A':>8} (only {len(values)}/{n_seeds} seeds)")

    # --- Per-repo breakdown (optional, for appendix) ---
    all_repos = set()
    for s in summaries:
        all_repos.update(s.get("per_repo", {}).keys())

    if all_repos:
        print(f"\n{'='*60}")
        print("Per-repo hit@1 (mean +/- std):")
        print(f"{'Repo':<35} {'Mean':>8} {'Std':>8}")
        print("-" * 55)

        repo_results = {}
        for repo in sorted(all_repos):
            values = []
            for s in summaries:
                repo_data = s.get("per_repo", {}).get(repo, {})
                val = repo_data.get("hit@1")
                if val is None:
                    val = repo_data.get("recall@1")
                if val is not None:
                    values.append(val)
            if len(values) == n_seeds:
                arr = np.array(values)
                mean = arr.mean()
                std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                print(f"{repo:<35} {mean:>8.2f} {std:>8.2f}")
                repo_results[repo] = {"mean": float(mean), "std": float(std)}

    # --- LaTeX-friendly output ---
    print(f"\n{'='*60}")
    print("LaTeX-ready (for paper table):")
    print(f"{'='*60}")
    parts = []
    for metric in METRICS:
        if metric in metric_results:
            m = metric_results[metric]
            parts.append(f"${m['mean']:.2f} \\pm {m['std']:.2f}$")
        else:
            parts.append("--")
    print(" & ".join(parts))

    # --- Save aggregated results ---
    output_path = os.path.join(args.base_dir, "multiseed_aggregated.json")
    aggregated = {
        "n_seeds": n_seeds,
        "summary_paths": summary_paths,
        "overall": metric_results,
    }
    try:
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nSaved aggregated results to: {output_path}")
    except OSError as e:
        print(f"\nCould not save to {output_path}: {e}")
        # Fall back to local dir
        fallback = "experiments/multiseed_aggregated.json"
        with open(fallback, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Saved to fallback: {fallback}")


if __name__ == "__main__":
    main()
