"""
Hardness stratification analysis for GREPO test set.

Buckets test examples by difficulty signals and computes R@1 per bucket
for our reranker vs BM25 baseline, showing where gains come from.

Signals:
  1. Number of GT files (1 vs 2+ files changed)
  2. GT file depth (shallow vs deep in directory tree)
  3. Stack trace / file path mentioned in issue text
  4. Repo size (number of Python files)
  5. Number of candidates (affects difficulty)
  6. BM25 rank of first GT file (easy if BM25 already found it)
"""

import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def recall_at_k(predicted, gt, k):
    gt_set = set(gt)
    found = sum(1 for p in predicted[:k] if p in gt_set)
    return found / len(gt_set) if gt_set else 0


def has_stack_trace(text):
    """Heuristic: issue text contains a stack trace."""
    patterns = [
        r'Traceback \(most recent',
        r'File ".*", line \d+',
        r'raise \w+Error',
        r'Error:.*\n.*at ',
        r'\.py:\d+',  # file.py:123
    ]
    return any(re.search(p, text) for p in patterns)


def has_file_mention(text, gt_files):
    """Heuristic: issue text mentions a GT file name or path."""
    for f in gt_files:
        fname = Path(f).name
        stem = Path(f).stem
        # Check if filename or stem appears in issue text
        if fname in text or stem in text:
            return True
        # Check partial path (last 2 components)
        parts = f.split('/')
        if len(parts) >= 2:
            partial = '/'.join(parts[-2:])
            if partial in text:
                return True
    return False


def bm25_rank_of_gt(bm25_list, gt_files):
    """Rank (1-indexed) of first GT file in BM25 results. Returns inf if not found."""
    gt_set = set(gt_files)
    for i, f in enumerate(bm25_list):
        if f in gt_set:
            return i + 1
    return float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",
                        default="experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl")
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl")
    parser.add_argument("--file_trees_dir", default="data/file_trees")
    parser.add_argument("--output", default="experiments/rankft_runB_graph/hardness_stratification.json")
    args = parser.parse_args()

    # Load test data (for issue text, timestamps)
    test_map = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], str(item.get("issue_id", "")))
            test_map[key] = item

    # Load predictions
    preds = []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))
    print(f"Loaded {len(preds)} predictions")

    # Load repo sizes (number of py files)
    repo_sizes = {}
    ft_dir = Path(args.file_trees_dir)
    if ft_dir.exists():
        for fpath in ft_dir.glob("*.json"):
            with open(fpath) as f:
                data = json.load(f)
            repo = data.get("repo", fpath.stem)
            n_py = data.get("num_py_files", len(data.get("py_files", [])))
            repo_sizes[repo] = n_py
    print(f"Repo sizes for {len(repo_sizes)} repos")

    # Stratification
    buckets = defaultdict(lambda: {"reranker": [], "bm25": [], "count": 0})

    for pred in preds:
        repo = pred["repo"]
        issue_id = str(pred.get("issue_id", ""))
        key = (repo, issue_id)
        td = test_map.get(key, {})
        issue_text = td.get("issue_text", "")

        gt = pred["ground_truth"]
        predicted = pred["predicted"]
        bm25 = pred["bm25_original"]

        r1_reranker = recall_at_k(predicted, gt, 1)
        r1_bm25 = recall_at_k(bm25, gt, 1)

        # --- Bucket 1: Number of GT files ---
        n_gt = len(gt)
        if n_gt == 1:
            b = "gt_count=1"
        elif n_gt == 2:
            b = "gt_count=2"
        else:
            b = "gt_count=3+"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

        # --- Bucket 2: File depth ---
        avg_depth = np.mean([f.count('/') for f in gt])
        if avg_depth <= 1:
            b = "depth<=1"
        elif avg_depth <= 2:
            b = "depth=2"
        else:
            b = "depth=3+"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

        # --- Bucket 3: Stack trace / file mention ---
        has_st = has_stack_trace(issue_text)
        has_fm = has_file_mention(issue_text, gt)
        if has_st:
            b = "has_stacktrace"
        elif has_fm:
            b = "has_file_mention"
        else:
            b = "no_hint"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

        # --- Bucket 4: Repo size ---
        n_files = repo_sizes.get(repo, 0)
        if n_files <= 500:
            b = "repo_small(<=500)"
        elif n_files <= 1000:
            b = "repo_medium(501-1000)"
        else:
            b = "repo_large(1000+)"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

        # --- Bucket 5: BM25 baseline rank of GT ---
        bm25_rank = bm25_rank_of_gt(bm25, gt)
        if bm25_rank <= 1:
            b = "bm25_rank=1"
        elif bm25_rank <= 5:
            b = "bm25_rank=2-5"
        elif bm25_rank <= 20:
            b = "bm25_rank=6-20"
        else:
            b = "bm25_rank=20+"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

        # --- Bucket 6: Number of candidates ---
        n_cands = pred.get("num_candidates", len(predicted))
        if n_cands <= 50:
            b = "cands<=50"
        elif n_cands <= 150:
            b = "cands=51-150"
        else:
            b = "cands=151+"
        buckets[b]["reranker"].append(r1_reranker)
        buckets[b]["bm25"].append(r1_bm25)
        buckets[b]["count"] += 1

    # Report
    print("\n" + "=" * 80)
    print("HARDNESS STRATIFICATION ANALYSIS")
    print("=" * 80)

    # Group by category
    categories = [
        ("GT File Count", ["gt_count=1", "gt_count=2", "gt_count=3+"]),
        ("File Depth", ["depth<=1", "depth=2", "depth=3+"]),
        ("Issue Hints", ["has_stacktrace", "has_file_mention", "no_hint"]),
        ("Repo Size", ["repo_small(<=500)", "repo_medium(501-1000)", "repo_large(1000+)"]),
        ("BM25 Baseline Rank", ["bm25_rank=1", "bm25_rank=2-5", "bm25_rank=6-20", "bm25_rank=20+"]),
        ("Candidate Pool Size", ["cands<=50", "cands=51-150", "cands=151+"]),
    ]

    results = {}
    for cat_name, bucket_names in categories:
        print(f"\n--- {cat_name} ---")
        print(f"  {'Bucket':>25} | {'N':>5} | {'Reranker R@1':>12} | {'BM25 R@1':>10} | {'Delta':>8}")
        print("  " + "-" * 70)
        cat_results = {}
        for bn in bucket_names:
            if bn not in buckets:
                continue
            b = buckets[bn]
            n = b["count"]
            r1_rr = np.mean(b["reranker"]) * 100
            r1_bm = np.mean(b["bm25"]) * 100
            delta = r1_rr - r1_bm
            print(f"  {bn:>25} | {n:>5} | {r1_rr:>11.2f}% | {r1_bm:>9.2f}% | {delta:>+7.2f}%")
            cat_results[bn] = {
                "n": n,
                "reranker_r1": round(r1_rr, 2),
                "bm25_r1": round(r1_bm, 2),
                "delta": round(delta, 2),
            }
        results[cat_name] = cat_results

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
