#!/usr/bin/env python3
"""Analyze path-overlap characteristics of different negative types.

Tests the hypothesis that BM25-hard negatives are "too easy" for a
path-semantic cross-encoder because they come from different parts
of the repository tree compared to ground-truth files.

Metrics computed per negative type (BM25-hard, Graph-hard, Random):
  1. % sharing any directory component with GT
  2. % sharing the immediate parent directory with GT
  3. Average path-component edit distance to nearest GT file
  4. Average Jaccard similarity of path components to nearest GT file
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Dict, List, Set, Tuple

import numpy as np

random.seed(42)
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────
BASE = "/home/chenlibin/grepo_agent"
TRAIN_DATA = os.path.join(BASE, "data/grepo_text/grepo_train.jsonl")
BM25_CANDIDATES = os.path.join(BASE, "data/rankft/grepo_train_bm25_top500.jsonl")
DEP_GRAPH_DIR = os.path.join(BASE, "data/dep_graphs")
FILE_TREE_DIR = os.path.join(BASE, "data/file_trees")

N_SAMPLES = 200  # number of training examples to sample (more = more stable)
N_NEG_PER_TYPE = 15  # negatives per type per example
TOP_BM25_HARD = 50  # consider top-k BM25 candidates (non-GT) as hard negs


# ── Utility functions ─────────────────────────────────────────────────

def path_components(p: str) -> List[str]:
    """Split a path into its directory + filename components."""
    return p.split("/")


def path_component_edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance on path components (dirs + filename)."""
    parts_a = path_components(a)
    parts_b = path_components(b)
    m, n = len(parts_a), len(parts_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cost = 0 if parts_a[i - 1] == parts_b[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[n]


def path_component_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the set of path components."""
    set_a = set(path_components(a))
    set_b = set(path_components(b))
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def shares_any_dir_component(neg: str, gt_files: Set[str]) -> bool:
    """Does negative share any directory component with any GT file?"""
    neg_dirs = set(PurePosixPath(neg).parts[:-1])  # exclude filename
    for gt in gt_files:
        gt_dirs = set(PurePosixPath(gt).parts[:-1])
        if neg_dirs & gt_dirs:
            return True
    return False


def shares_parent_dir(neg: str, gt_files: Set[str]) -> bool:
    """Does negative share the immediate parent directory with any GT file?"""
    neg_parent = str(PurePosixPath(neg).parent)
    for gt in gt_files:
        if str(PurePosixPath(gt).parent) == neg_parent:
            return True
    return False


def shares_top2_dir(neg: str, gt_files: Set[str]) -> bool:
    """Does negative share the top-level package directory with any GT file?
    E.g., cirq/ops/foo.py and cirq/linalg/bar.py both have top-level 'cirq'.
    We check the first 2 path components match.
    """
    neg_parts = PurePosixPath(neg).parts
    neg_top2 = neg_parts[:min(2, len(neg_parts))]
    for gt in gt_files:
        gt_parts = PurePosixPath(gt).parts
        gt_top2 = gt_parts[:min(2, len(gt_parts))]
        if neg_top2 == gt_top2:
            return True
    return False


def min_edit_distance_to_gt(neg: str, gt_files: Set[str]) -> int:
    """Minimum path-component edit distance from neg to any GT file."""
    return min(path_component_edit_distance(neg, gt) for gt in gt_files)


def max_jaccard_to_gt(neg: str, gt_files: Set[str]) -> float:
    """Maximum path-component Jaccard similarity from neg to any GT file."""
    return max(path_component_jaccard(neg, gt) for gt in gt_files)


def avg_dir_depth_distance(neg: str, gt_files: Set[str]) -> float:
    """Average absolute difference in directory depth."""
    neg_depth = len(PurePosixPath(neg).parts) - 1
    return np.mean([abs(neg_depth - (len(PurePosixPath(gt).parts) - 1))
                     for gt in gt_files])


# ── Build indices ─────────────────────────────────────────────────────

def build_cochange_index(train_data_path: str, min_cochange: int = 1):
    repo_pairs = defaultdict(Counter)
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            repo = item["repo"]
            files = sorted(item.get("changed_py_files", []))
            for i, fa in enumerate(files):
                for fb in files[i + 1:]:
                    repo_pairs[repo][(fa, fb)] += 1

    index = defaultdict(lambda: defaultdict(set))
    for repo, pairs in repo_pairs.items():
        for (fa, fb), count in pairs.items():
            if count >= min_cochange:
                index[repo][fa].add(fb)
                index[repo][fb].add(fa)
    return dict(index)


def build_import_adjacency(dep_graph_dir: str):
    result = defaultdict(lambda: defaultdict(set))
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        for importer, imported_list in rels.get("file_imports", {}).items():
            for imported in imported_list:
                if importer.endswith(".py") and imported.endswith(".py"):
                    result[repo][importer].add(imported)
                    result[repo][imported].add(importer)
    return dict(result)


def load_file_trees(file_tree_dir: str):
    repo_files = {}
    for fname in os.listdir(file_tree_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(file_tree_dir, fname)) as f:
            tree = json.load(f)
        repo = tree.get("repo", fname.replace(".json", ""))
        repo_files[repo] = tree.get("py_files", [])
    return repo_files


def load_bm25_candidates(bm25_path: str):
    """Load BM25 candidates indexed by (repo, issue_id)."""
    result = {}
    with open(bm25_path) as f:
        for line in f:
            item = json.loads(line)
            key = (item["repo"], item["issue_id"])
            result[key] = item.get("bm25_candidates", [])
    return result


def load_train_data(train_path: str):
    data = []
    with open(train_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ── Main analysis ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PATH-OVERLAP ANALYSIS: BM25 vs Graph vs Random NEGATIVES")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_data = load_train_data(TRAIN_DATA)
    print(f"  Training examples: {len(train_data)}")

    bm25_cands = load_bm25_candidates(BM25_CANDIDATES)
    print(f"  BM25 candidate entries: {len(bm25_cands)}")

    cochange_index = build_cochange_index(TRAIN_DATA)
    print(f"  Co-change index: {len(cochange_index)} repos")

    import_index = build_import_adjacency(DEP_GRAPH_DIR)
    print(f"  Import index: {len(import_index)} repos")

    repo_files = load_file_trees(FILE_TREE_DIR)
    print(f"  File trees: {len(repo_files)} repos")

    # Filter examples that have BM25 candidates AND py files in GT
    eligible = []
    for item in train_data:
        repo = item["repo"]
        iid = item["issue_id"]
        gt = set(item.get("changed_py_files", []))
        if not gt:
            continue
        bm25 = bm25_cands.get((repo, iid), [])
        bm25_negs = [c for c in bm25 if c not in gt]
        all_files = repo_files.get(repo, [])
        if len(bm25_negs) >= 5 and len(all_files) >= 20:
            eligible.append(item)

    print(f"  Eligible examples (have BM25 negs & repo files): {len(eligible)}")

    # Sample
    sampled = random.sample(eligible, min(N_SAMPLES, len(eligible)))
    print(f"  Sampled for analysis: {len(sampled)}")

    # Collect metrics per negative type
    metrics = {
        "bm25_hard": {
            "shares_any_dir": [], "shares_parent_dir": [], "shares_top2_dir": [],
            "edit_dist": [], "jaccard": [], "depth_dist": [],
            "n_available": [],
        },
        "graph_hard": {
            "shares_any_dir": [], "shares_parent_dir": [], "shares_top2_dir": [],
            "edit_dist": [], "jaccard": [], "depth_dist": [],
            "n_available": [],
        },
        "random": {
            "shares_any_dir": [], "shares_parent_dir": [], "shares_top2_dir": [],
            "edit_dist": [], "jaccard": [], "depth_dist": [],
            "n_available": [],
        },
    }

    skipped_graph = 0
    for item in sampled:
        repo = item["repo"]
        iid = item["issue_id"]
        gt = set(item.get("changed_py_files", []))
        all_files = repo_files.get(repo, [])
        all_files_set = set(all_files)

        # ── BM25-hard negatives ──
        bm25 = bm25_cands.get((repo, iid), [])
        bm25_negs = [c for c in bm25 if c not in gt][:TOP_BM25_HARD]
        bm25_sample = bm25_negs[:N_NEG_PER_TYPE]
        metrics["bm25_hard"]["n_available"].append(len(bm25_negs))

        for neg in bm25_sample:
            metrics["bm25_hard"]["shares_any_dir"].append(shares_any_dir_component(neg, gt))
            metrics["bm25_hard"]["shares_parent_dir"].append(shares_parent_dir(neg, gt))
            metrics["bm25_hard"]["shares_top2_dir"].append(shares_top2_dir(neg, gt))
            metrics["bm25_hard"]["edit_dist"].append(min_edit_distance_to_gt(neg, gt))
            metrics["bm25_hard"]["jaccard"].append(max_jaccard_to_gt(neg, gt))
            metrics["bm25_hard"]["depth_dist"].append(avg_dir_depth_distance(neg, gt))

        # ── Graph-hard negatives (import + co-change neighbors) ──
        graph_pool = set()
        repo_cc = cochange_index.get(repo, {})
        repo_imp = import_index.get(repo, {})
        for gt_f in gt:
            graph_pool.update(repo_cc.get(gt_f, set()))
            graph_pool.update(repo_imp.get(gt_f, set()))
        graph_pool -= gt
        # Also filter to files that exist in repo file tree
        graph_pool &= all_files_set
        graph_list = sorted(graph_pool)
        random.shuffle(graph_list)
        graph_sample = graph_list[:N_NEG_PER_TYPE]
        metrics["graph_hard"]["n_available"].append(len(graph_pool))

        if not graph_sample:
            skipped_graph += 1

        for neg in graph_sample:
            metrics["graph_hard"]["shares_any_dir"].append(shares_any_dir_component(neg, gt))
            metrics["graph_hard"]["shares_parent_dir"].append(shares_parent_dir(neg, gt))
            metrics["graph_hard"]["shares_top2_dir"].append(shares_top2_dir(neg, gt))
            metrics["graph_hard"]["edit_dist"].append(min_edit_distance_to_gt(neg, gt))
            metrics["graph_hard"]["jaccard"].append(max_jaccard_to_gt(neg, gt))
            metrics["graph_hard"]["depth_dist"].append(avg_dir_depth_distance(neg, gt))

        # ── Random negatives ──
        random_pool = [f for f in all_files if f not in gt]
        random.shuffle(random_pool)
        random_sample = random_pool[:N_NEG_PER_TYPE]
        metrics["random"]["n_available"].append(len(random_pool))

        for neg in random_sample:
            metrics["random"]["shares_any_dir"].append(shares_any_dir_component(neg, gt))
            metrics["random"]["shares_parent_dir"].append(shares_parent_dir(neg, gt))
            metrics["random"]["shares_top2_dir"].append(shares_top2_dir(neg, gt))
            metrics["random"]["edit_dist"].append(min_edit_distance_to_gt(neg, gt))
            metrics["random"]["jaccard"].append(max_jaccard_to_gt(neg, gt))
            metrics["random"]["depth_dist"].append(avg_dir_depth_distance(neg, gt))

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n  (Skipped {skipped_graph}/{len(sampled)} examples with no graph neighbors)")
    print("\n" + "=" * 70)
    print("RESULTS: Path-Overlap Metrics by Negative Type")
    print("=" * 70)

    for neg_type in ["bm25_hard", "graph_hard", "random"]:
        m = metrics[neg_type]
        n = len(m["shares_any_dir"])
        print(f"\n{'─' * 50}")
        print(f"  {neg_type.upper()} (n={n} neg files across {len(sampled)} examples)")
        print(f"  Avg available pool size: {np.mean(m['n_available']):.1f}")
        print(f"{'─' * 50}")
        if n == 0:
            print("  (No samples available)")
            continue

        share_any = np.mean(m["shares_any_dir"]) * 100
        share_parent = np.mean(m["shares_parent_dir"]) * 100
        share_top2 = np.mean(m["shares_top2_dir"]) * 100
        avg_ed = np.mean(m["edit_dist"])
        avg_jac = np.mean(m["jaccard"])
        avg_depth = np.mean(m["depth_dist"])

        # Bootstrap 95% CI for share_any_dir
        bootstrap_vals = []
        arr = np.array(m["shares_any_dir"], dtype=float)
        for _ in range(1000):
            idx = np.random.choice(len(arr), size=len(arr), replace=True)
            bootstrap_vals.append(arr[idx].mean())
        ci_lo, ci_hi = np.percentile(bootstrap_vals, [2.5, 97.5])

        print(f"  % sharing ANY dir component with GT:    {share_any:6.1f}%  (95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])")
        print(f"  % sharing PARENT dir with GT:           {share_parent:6.1f}%")
        print(f"  % sharing top-2 package dir with GT:    {share_top2:6.1f}%")
        print(f"  Avg path-component edit distance to GT: {avg_ed:6.2f}")
        print(f"  Avg path-component Jaccard with GT:     {avg_jac:6.3f}")
        print(f"  Avg dir depth distance to GT:           {avg_depth:6.2f}")

    # ── Detailed distribution of edit distances ──
    print("\n" + "=" * 70)
    print("DISTRIBUTION: Path-Component Edit Distance (histogram)")
    print("=" * 70)

    for neg_type in ["bm25_hard", "graph_hard", "random"]:
        m = metrics[neg_type]
        if not m["edit_dist"]:
            continue
        arr = np.array(m["edit_dist"])
        print(f"\n  {neg_type.upper()}:")
        print(f"    Mean={arr.mean():.2f}  Median={np.median(arr):.1f}  "
              f"Std={arr.std():.2f}  Min={arr.min()}  Max={arr.max()}")
        # Histogram
        bins = list(range(0, min(int(arr.max()) + 2, 12)))
        counts, _ = np.histogram(arr, bins=bins)
        total = len(arr)
        for i in range(len(counts)):
            bar = "#" * int(counts[i] / total * 60)
            print(f"    [{bins[i]:2d}-{bins[i+1]:2d}): {counts[i]:5d} ({counts[i]/total*100:5.1f}%) {bar}")

    # ── Statistical comparison ──
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: BM25 vs Graph vs Random")
    print("=" * 70)

    from scipy import stats

    for metric_name, higher_is_closer in [
        ("shares_any_dir", True),
        ("shares_parent_dir", True),
        ("edit_dist", False),
        ("jaccard", True),
    ]:
        print(f"\n  Metric: {metric_name}")
        for pair in [("bm25_hard", "random"), ("graph_hard", "random"), ("bm25_hard", "graph_hard")]:
            a_vals = np.array(metrics[pair[0]][metric_name], dtype=float)
            b_vals = np.array(metrics[pair[1]][metric_name], dtype=float)
            if len(a_vals) == 0 or len(b_vals) == 0:
                print(f"    {pair[0]} vs {pair[1]}: insufficient data")
                continue
            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            print(f"    {pair[0]:12s} (mean={a_vals.mean():.3f}) vs "
                  f"{pair[1]:12s} (mean={b_vals.mean():.3f})  "
                  f"U={stat:.0f}  p={pval:.4e}")

    # ── Concrete examples ──
    print("\n" + "=" * 70)
    print("CONCRETE EXAMPLES: BM25-hard vs Graph-hard vs GT paths")
    print("=" * 70)

    n_examples_shown = 0
    for item in sampled[:30]:
        repo = item["repo"]
        iid = item["issue_id"]
        gt = set(item.get("changed_py_files", []))

        bm25 = bm25_cands.get((repo, iid), [])
        bm25_negs = [c for c in bm25 if c not in gt][:5]

        graph_pool = set()
        repo_cc = cochange_index.get(repo, {})
        repo_imp = import_index.get(repo, {})
        for gt_f in gt:
            graph_pool.update(repo_cc.get(gt_f, set()))
            graph_pool.update(repo_imp.get(gt_f, set()))
        graph_pool -= gt
        graph_pool &= set(repo_files.get(repo, []))
        graph_list = sorted(graph_pool)[:5]

        if not graph_list:
            continue

        n_examples_shown += 1
        if n_examples_shown > 5:
            break

        print(f"\n  [{repo}] Issue #{iid}")
        print(f"  GT files:")
        for g in sorted(gt):
            print(f"    + {g}")
        print(f"  BM25-hard negatives (top 5):")
        for neg in bm25_negs:
            ed = min_edit_distance_to_gt(neg, gt)
            jac = max_jaccard_to_gt(neg, gt)
            sdir = shares_parent_dir(neg, gt)
            print(f"    - {neg}  (ed={ed}, jac={jac:.2f}, same_parent={sdir})")
        print(f"  Graph-hard negatives (top 5):")
        for neg in graph_list:
            ed = min_edit_distance_to_gt(neg, gt)
            jac = max_jaccard_to_gt(neg, gt)
            sdir = shares_parent_dir(neg, gt)
            print(f"    - {neg}  (ed={ed}, jac={jac:.2f}, same_parent={sdir})")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    bm25_any = np.mean(metrics["bm25_hard"]["shares_any_dir"]) * 100
    graph_any = np.mean(metrics["graph_hard"]["shares_any_dir"]) * 100
    rand_any = np.mean(metrics["random"]["shares_any_dir"]) * 100

    bm25_parent = np.mean(metrics["bm25_hard"]["shares_parent_dir"]) * 100
    graph_parent = np.mean(metrics["graph_hard"]["shares_parent_dir"]) * 100 if metrics["graph_hard"]["shares_parent_dir"] else 0
    rand_parent = np.mean(metrics["random"]["shares_parent_dir"]) * 100

    bm25_ed = np.mean(metrics["bm25_hard"]["edit_dist"])
    graph_ed = np.mean(metrics["graph_hard"]["edit_dist"]) if metrics["graph_hard"]["edit_dist"] else 0
    rand_ed = np.mean(metrics["random"]["edit_dist"])

    bm25_jac = np.mean(metrics["bm25_hard"]["jaccard"])
    graph_jac = np.mean(metrics["graph_hard"]["jaccard"]) if metrics["graph_hard"]["jaccard"] else 0
    rand_jac = np.mean(metrics["random"]["jaccard"])

    print(f"""
    Metric                          BM25-hard    Graph-hard    Random
    ──────────────────────────────  ──────────   ──────────   ──────────
    % shares ANY dir with GT        {bm25_any:6.1f}%      {graph_any:6.1f}%      {rand_any:6.1f}%
    % shares PARENT dir with GT     {bm25_parent:6.1f}%      {graph_parent:6.1f}%      {rand_parent:6.1f}%
    Avg edit distance to GT         {bm25_ed:6.2f}        {graph_ed:6.2f}        {rand_ed:6.2f}
    Avg Jaccard with GT             {bm25_jac:6.3f}       {graph_jac:6.3f}       {rand_jac:6.3f}
    """)

    if bm25_any < rand_any + 5 and bm25_ed > graph_ed * 0.9:
        print("  CONCLUSION: Hypothesis SUPPORTED.")
        print("  BM25-hard negatives have similar or lower path overlap with GT")
        print("  compared to random negatives, and substantially lower than graph-hard.")
        print("  A path-semantic model can trivially reject BM25 negs based on path alone.")
    elif bm25_any > graph_any:
        print("  CONCLUSION: Hypothesis NOT clearly supported.")
        print("  BM25-hard negatives actually have higher path overlap than graph-hard.")
    else:
        print("  CONCLUSION: Results are mixed. See detailed metrics above.")


if __name__ == "__main__":
    main()
