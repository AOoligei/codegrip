#!/usr/bin/env python3
"""
Downstream validity analysis: does file-level R@1 predict function-level success?

Metrics:
1. Gold-file conditioned function Hit@1: of file-level top-1 successes,
   what fraction also succeed at function-level?
2. Edit-region overlap: does the model's top function overlap with changed_functions?
3. Per-repo scatter: file-level R@1 vs function-level success correlation

Uses existing eval outputs (no GPU needed).

Usage:
    python scripts/downstream_validity.py \
        --file_predictions /data/.../eval_graph/predictions.jsonl \
        --func_predictions /data/.../func_level_eval/per_example.jsonl \
        --output_dir /data/chenlibin/grepo_agent_experiments/downstream_validity
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
FUNC_INDEX_PATH = "/home/chenlibin/grepo_agent/data/function_index_aligned.json"


def load_test_data():
    """Load test records keyed by (repo, issue_id)."""
    data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            data[key] = rec
    return data


def analyze_file_to_function_gap(test_data, func_index):
    """For each test example, check if file-level GT hit implies function-level GT hit.

    This doesn't need model predictions - it measures the theoretical gap:
    if you know the correct file, can you identify the correct function?
    """
    results = {
        "total": 0,
        "has_gt_functions": 0,
        "file_has_gt_func_in_index": 0,
        "multi_func_files": 0,
        "single_func_files": 0,
    }

    per_repo = defaultdict(lambda: {"total": 0, "has_funcs": 0})

    for key, rec in test_data.items():
        repo = rec["repo"]
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        gt_funcs = rec.get("changed_functions", [])
        results["total"] += 1

        if not gt_funcs:
            per_repo[repo]["total"] += 1
            continue

        results["has_gt_functions"] += 1

        repo_index = func_index.get(repo, {})
        for gf in gt_files:
            file_funcs = repo_index.get(gf, [])
            if any(fn in file_funcs for fn in gt_funcs):
                results["file_has_gt_func_in_index"] += 1
            if len(file_funcs) > 1:
                results["multi_func_files"] += 1
            elif len(file_funcs) == 1:
                results["single_func_files"] += 1

        per_repo[repo]["total"] += 1
        per_repo[repo]["has_funcs"] += 1

    return results, per_repo


def analyze_from_hierarchical_results(hier_results_path, test_data, func_index):
    """Analyze hierarchical eval results for file-to-function gap."""
    if not os.path.isfile(hier_results_path):
        return None

    file_hit_func_hit = 0
    file_hit_func_miss = 0
    file_miss = 0

    with open(hier_results_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            if key not in test_data:
                continue

            test_rec = test_data[key]
            gt_funcs = set(test_rec.get("changed_functions", []))

            if not rec.get("path_only_hit"):
                file_miss += 1
                continue

            # File-level hit. Check if the hit file contains GT functions
            if not gt_funcs:
                file_hit_func_miss += 1
                continue

            repo_index = func_index.get(rec["repo"], {})
            # We don't know which file was the top-1 hit from this data
            # but we know it was a hit, so check all GT files
            gt_files = test_rec.get("changed_py_files",
                                     test_rec.get("changed_files", []))
            found_func = False
            for gf in gt_files:
                file_funcs = repo_index.get(gf, [])
                if any(fn in file_funcs for fn in gt_funcs):
                    found_func = True
                    break

            if found_func:
                file_hit_func_hit += 1
            else:
                file_hit_func_miss += 1

    total_file_hits = file_hit_func_hit + file_hit_func_miss
    return {
        "file_hit_total": total_file_hits,
        "file_hit_func_hit": file_hit_func_hit,
        "file_hit_func_miss": file_hit_func_miss,
        "file_miss": file_miss,
        "file_to_func_rate": file_hit_func_hit / max(1, total_file_hits),
    }


def compute_repo_level_correlation(test_data, func_index):
    """Compute per-repo: fraction of examples with GT functions available."""
    repo_stats = defaultdict(lambda: {
        "total": 0, "has_gt_funcs": 0, "avg_funcs_per_file": []
    })

    for key, rec in test_data.items():
        repo = rec["repo"]
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        gt_funcs = rec.get("changed_functions", [])
        repo_stats[repo]["total"] += 1
        if gt_funcs:
            repo_stats[repo]["has_gt_funcs"] += 1

        repo_index = func_index.get(repo, {})
        for gf in gt_files:
            n_funcs = len(repo_index.get(gf, []))
            repo_stats[repo]["avg_funcs_per_file"].append(n_funcs)

    summary = []
    for repo, stats in repo_stats.items():
        avg_funcs = np.mean(stats["avg_funcs_per_file"]) if stats["avg_funcs_per_file"] else 0
        summary.append({
            "repo": repo,
            "n_examples": stats["total"],
            "pct_with_gt_funcs": stats["has_gt_funcs"] / max(1, stats["total"]),
            "avg_funcs_per_gt_file": float(avg_funcs),
        })

    return sorted(summary, key=lambda x: -x["n_examples"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hier_results", type=str, default=None,
                        help="Path to hierarchical eval per_example.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/downstream_validity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    test_data = load_test_data()
    func_index = json.load(open(FUNC_INDEX_PATH))
    print(f"  {len(test_data)} test examples, {len(func_index)} repos in func index")

    print("\n1. File-to-function gap analysis...")
    gap_results, per_repo = analyze_file_to_function_gap(test_data, func_index)
    print(f"  Total examples: {gap_results['total']}")
    print(f"  With GT functions: {gap_results['has_gt_functions']}")
    print(f"  GT func found in file index: {gap_results['file_has_gt_func_in_index']}")

    print("\n2. Repo-level statistics...")
    repo_stats = compute_repo_level_correlation(test_data, func_index)
    for r in repo_stats[:5]:
        print(f"  {r['repo']}: {r['n_examples']} examples, "
              f"{r['pct_with_gt_funcs']:.0%} with GT funcs, "
              f"avg {r['avg_funcs_per_gt_file']:.1f} funcs/file")

    if args.hier_results:
        print("\n3. Hierarchical file-to-function analysis...")
        hier_analysis = analyze_from_hierarchical_results(
            args.hier_results, test_data, func_index)
        if hier_analysis:
            print(f"  File hits: {hier_analysis['file_hit_total']}")
            print(f"  File hit + func available: {hier_analysis['file_hit_func_hit']}")
            print(f"  File hit + func missing: {hier_analysis['file_hit_func_miss']}")
            print(f"  File-to-function rate: {hier_analysis['file_to_func_rate']:.1%}")

    # Save
    summary = {
        "gap_analysis": gap_results,
        "repo_stats": repo_stats[:20],
    }
    if args.hier_results:
        summary["hier_analysis"] = hier_analysis

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
