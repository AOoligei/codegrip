#!/usr/bin/env python3
"""
Expansion weight sensitivity analysis for the CodeGRIP paper.

Answers reviewer question: "How sensitive are the expansion weights
(lambda_cc, lambda_imp) and thresholds (tau_cc) to changes?"

Varies ONE parameter at a time from default settings:
  - Co-change threshold (tau_cc): 0.01, 0.03, *0.05*, 0.10, 0.20
  - Max co-change neighbors:      3, 5, *10*, 15, 20
  - Max import neighbors:         3, 5, *10*, 15, 20

For each setting computes:
  - Average pool size
  - Oracle recall (GT in pool)
  - Number of examples affected (pool differs from BM25-only)

CPU-only. No GPU needed.

Usage:
    python scripts/sensitivity_expansion.py
    python scripts/sensitivity_expansion.py --save_interesting
"""

import json
import os
import sys
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BM25_CANDIDATES = os.path.join(
    PROJECT_ROOT, "data", "rankft", "bm25_top_matched_candidates.jsonl"
)
TRAIN_DATA = os.path.join(
    PROJECT_ROOT, "data", "grepo_text", "grepo_train.jsonl"
)
TEST_DATA = os.path.join(
    PROJECT_ROOT, "data", "grepo_text", "grepo_test.jsonl"
)
DEP_GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "dep_graphs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "rankft")

# Default expansion parameters (from create_edge_type_pools.py / graph_expansion.py)
DEFAULT_MIN_COCHANGE_SCORE = 0.05
DEFAULT_MAX_COCHANGE = 10
DEFAULT_MAX_IMPORT = 10
SEED_SIZE = 20  # top-K BM25 candidates used as seed


# ============================================================
# Index builders (same logic as src/eval/graph_expansion.py)
# ============================================================

def build_cochange_index(
    train_data_path: str, min_cochange: int = 1
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build per-repo co-change index from training PR data."""
    repo_cochanges: Dict[str, Counter] = defaultdict(Counter)
    repo_file_count: Dict[str, Counter] = defaultdict(Counter)

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") != "train":
                continue
            repo = item["repo"]
            files = item.get("changed_py_files", [])
            if not files:
                files = [
                    fp for fp in item.get("changed_files", [])
                    if fp.endswith(".py")
                ]
            for fp in files:
                repo_file_count[repo][fp] += 1
            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score
    return index


def build_import_index(
    dep_graph_dir: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo bidirectional import index."""
    index: Dict[str, Dict[str, Set[str]]] = {}
    if not os.path.isdir(dep_graph_dir):
        return index

    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)

        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for src, targets in rels.get("file_imports", {}).items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)
        for src_func, callees in rels.get("call_graph", {}).items():
            src_file = src_func.split(":")[0] if ":" in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(":")[0] if ":" in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)
        index[repo] = dict(neighbors)
    return index


# ============================================================
# Expansion logic (mirrors create_edge_type_pools.py exactly)
# ============================================================

def expand_with_cochange(
    seed_files: List[str],
    repo_cc: Dict[str, Dict[str, float]],
    max_expand: int = 10,
    min_score: float = 0.05,
) -> List[str]:
    """Get co-change neighbors for seed files, sorted by score desc."""
    seed_set = set(seed_files)
    cands: Dict[str, float] = {}
    for pred_file in seed_files:
        for neighbor, score in repo_cc.get(pred_file, {}).items():
            if neighbor not in seed_set and score >= min_score:
                cands[neighbor] = max(cands.get(neighbor, 0), score)
    sorted_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_cands[:max_expand]]


def expand_with_imports(
    seed_files: List[str],
    repo_imp: Dict[str, Set[str]],
    exclude: Set[str],
    max_expand: int = 10,
) -> List[str]:
    """Get import neighbors for seed files, scored by link count."""
    scores: Dict[str, int] = defaultdict(int)
    for pred_file in seed_files:
        for neighbor in repo_imp.get(pred_file, set()):
            if neighbor not in exclude:
                scores[neighbor] += 1
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_cands[:max_expand]]


def build_expanded_pool(
    bm25_candidates: List[str],
    repo: str,
    cc_index: Dict[str, Dict[str, Dict[str, float]]],
    imp_index: Dict[str, Dict[str, Set[str]]],
    max_cochange: int = 10,
    max_import: int = 10,
    min_cochange_score: float = 0.05,
) -> Tuple[List[str], int, int]:
    """
    Build expanded candidate pool for a single example.
    Returns (merged_candidates, n_cc_added, n_imp_added).
    """
    seed = bm25_candidates[:SEED_SIZE]

    repo_cc = cc_index.get(repo, {})
    cc_expansion = expand_with_cochange(
        seed, repo_cc, max_expand=max_cochange, min_score=min_cochange_score
    )

    already = set(seed) | set(cc_expansion)
    repo_imp = imp_index.get(repo, {})
    imp_expansion = expand_with_imports(
        seed, repo_imp, exclude=already, max_expand=max_import
    )

    # Graph neighbors first, then BM25 fill (deduped), truncate
    graph_neighbors = cc_expansion + imp_expansion
    seen = set()
    merged = []
    for c in graph_neighbors:
        if c not in seen:
            merged.append(c)
            seen.add(c)
    for c in bm25_candidates:
        if c not in seen:
            merged.append(c)
            seen.add(c)
    merged = merged[:len(bm25_candidates)]

    return merged, len(cc_expansion), len(imp_expansion)


# ============================================================
# Oracle recall
# ============================================================

def oracle_recall(candidates: List[str], gt: Set[str]) -> float:
    """Fraction of GT files appearing anywhere in candidate pool (%)."""
    if not gt:
        return 0.0
    return len(set(candidates) & gt) / len(gt) * 100.0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Expansion parameter sensitivity analysis"
    )
    parser.add_argument(
        "--save_interesting", action="store_true",
        help="Save candidate pool files for 2-3 interesting settings",
    )
    args = parser.parse_args()

    # ---- Load data ----
    print("Loading BM25 candidates...")
    bm25_data = {}
    with open(BM25_CANDIDATES) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            bm25_data[key] = d["candidates"]
    print(f"  {len(bm25_data)} examples")

    print("Loading test ground truth...")
    gt_data = {}
    with open(TEST_DATA) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            gt = set(d.get("changed_py_files", []))
            if gt:
                gt_data[key] = gt
    print(f"  {len(gt_data)} examples with GT")

    print("Building co-change index...")
    cc_index = build_cochange_index(TRAIN_DATA, min_cochange=1)
    print(f"  {len(cc_index)} repos")

    print("Building import index...")
    imp_index = build_import_index(DEP_GRAPH_DIR)
    print(f"  {len(imp_index)} repos")

    # ---- BM25-only baseline ----
    bm25_recalls = []
    bm25_sizes = []
    for key in sorted(gt_data.keys()):
        if key in bm25_data:
            bm25_recalls.append(oracle_recall(bm25_data[key], gt_data[key]))
            bm25_sizes.append(len(bm25_data[key]))
    bm25_oracle = np.mean(bm25_recalls) if bm25_recalls else 0.0
    bm25_avg_pool = np.mean(bm25_sizes) if bm25_sizes else 0.0
    print(f"\nBM25-only baseline: oracle recall={bm25_oracle:.2f}%, "
          f"avg pool={bm25_avg_pool:.1f}, n={len(bm25_recalls)}")

    # ---- Define parameter sweep ----
    # Each sweep varies ONE parameter, others held at default.
    sweeps = {
        "tau_cc (co-change threshold)": {
            "param_name": "min_cochange_score",
            "values": [0.01, 0.03, 0.05, 0.10, 0.20],
            "default": DEFAULT_MIN_COCHANGE_SCORE,
            "fixed": {
                "max_cochange": DEFAULT_MAX_COCHANGE,
                "max_import": DEFAULT_MAX_IMPORT,
            },
        },
        "max_cc (co-change neighbors)": {
            "param_name": "max_cochange",
            "values": [3, 5, 10, 15, 20],
            "default": DEFAULT_MAX_COCHANGE,
            "fixed": {
                "min_cochange_score": DEFAULT_MIN_COCHANGE_SCORE,
                "max_import": DEFAULT_MAX_IMPORT,
            },
        },
        "max_imp (import neighbors)": {
            "param_name": "max_import",
            "values": [3, 5, 10, 15, 20],
            "default": DEFAULT_MAX_IMPORT,
            "fixed": {
                "min_cochange_score": DEFAULT_MIN_COCHANGE_SCORE,
                "max_cochange": DEFAULT_MAX_COCHANGE,
            },
        },
    }

    # ---- Run sweeps ----
    all_results = {}  # (sweep_name, param_value) -> {metrics dict}
    saved_pools = {}  # for --save_interesting

    for sweep_name, sweep_cfg in sweeps.items():
        print(f"\n{'=' * 70}")
        print(f"  Sweep: {sweep_name}")
        print(f"{'=' * 70}")

        param_name = sweep_cfg["param_name"]
        fixed = sweep_cfg["fixed"]
        default_val = sweep_cfg["default"]

        header = (f"{'Value':>8}  {'Oracle%':>8}  {'Delta':>7}  "
                  f"{'AvgPool':>8}  {'AvgCC':>6}  {'AvgImp':>6}  "
                  f"{'Affected':>8}  {'N':>5}")
        print(header)
        print("-" * len(header))

        for val in sweep_cfg["values"]:
            # Build params dict
            params = dict(fixed)
            params[param_name] = val

            recalls = []
            pool_sizes = []
            cc_counts = []
            imp_counts = []
            n_affected = 0
            pool_data = []

            for key in sorted(gt_data.keys()):
                if key not in bm25_data:
                    continue
                repo, issue_id = key
                bm25_cands = bm25_data[key]

                merged, n_cc, n_imp = build_expanded_pool(
                    bm25_cands, repo, cc_index, imp_index,
                    max_cochange=params["max_cochange"],
                    max_import=params["max_import"],
                    min_cochange_score=params["min_cochange_score"],
                )

                gt = gt_data[key]
                recalls.append(oracle_recall(merged, gt))
                pool_sizes.append(len(merged))
                cc_counts.append(n_cc)
                imp_counts.append(n_imp)

                bm25_set = set(bm25_cands)
                if any(c not in bm25_set for c in merged):
                    n_affected += 1

                pool_data.append({
                    "repo": repo,
                    "issue_id": issue_id,
                    "candidates": merged,
                })

            avg_recall = np.mean(recalls)
            delta = avg_recall - bm25_oracle
            avg_pool = np.mean(pool_sizes)
            avg_cc = np.mean(cc_counts)
            avg_imp = np.mean(imp_counts)
            n = len(recalls)

            is_default = (val == default_val)
            marker = " *" if is_default else ""

            print(f"{val:>8}  {avg_recall:>7.2f}%  {delta:>+6.2f}%  "
                  f"{avg_pool:>8.1f}  {avg_cc:>5.1f}  {avg_imp:>5.1f}  "
                  f"{n_affected:>8}  {n:>5}{marker}")

            all_results[(sweep_name, val)] = {
                "oracle_recall": avg_recall,
                "delta_vs_bm25": delta,
                "avg_pool_size": avg_pool,
                "avg_cc_neighbors": avg_cc,
                "avg_imp_neighbors": avg_imp,
                "n_affected": n_affected,
                "n_examples": n,
                "is_default": is_default,
                "pool_data": pool_data,
            }

        print(f"  (* = default setting)")

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print("SUMMARY: Sensitivity of Graph Expansion Parameters")
    print(f"{'=' * 70}")
    print(f"BM25-only baseline: oracle recall = {bm25_oracle:.2f}%\n")

    for sweep_name, sweep_cfg in sweeps.items():
        vals = sweep_cfg["values"]
        recalls = [all_results[(sweep_name, v)]["oracle_recall"] for v in vals]
        recall_range = max(recalls) - min(recalls)
        best_val = vals[np.argmax(recalls)]
        worst_val = vals[np.argmin(recalls)]
        print(f"{sweep_name}:")
        print(f"  Range: {min(recalls):.2f}% - {max(recalls):.2f}%  "
              f"(spread = {recall_range:.2f}pp)")
        print(f"  Best:  {best_val} ({max(recalls):.2f}%), "
              f"Worst: {worst_val} ({min(recalls):.2f}%)")
        # All above BM25?
        all_above = all(r > bm25_oracle for r in recalls)
        print(f"  All settings improve over BM25-only: {all_above}")
        print()

    # ---- Identify and save interesting settings ----
    if args.save_interesting:
        print(f"\n{'=' * 70}")
        print("Saving candidate pools for interesting settings...")
        print(f"{'=' * 70}")

        # Strategy: save the best non-default from each sweep dimension,
        # picking diverse settings. Max 3 saved files.
        interesting = []
        used_keys = set()

        # 1. Best overall oracle recall across all sweeps (non-default)
        best_key = None
        best_recall = -1.0
        for (sname, val), res in all_results.items():
            if not res["is_default"] and res["oracle_recall"] > best_recall:
                best_recall = res["oracle_recall"]
                best_key = (sname, val)
        if best_key:
            interesting.append(("best_overall", best_key))
            used_keys.add(best_key)

        # 2. Most conservative (smallest expansion that still helps)
        conservative_key = None
        min_expansion = float("inf")
        for (sname, val), res in all_results.items():
            if (sname, val) not in used_keys and not res["is_default"]:
                if res["delta_vs_bm25"] > 0:
                    total_exp = res["avg_cc_neighbors"] + res["avg_imp_neighbors"]
                    if total_exp < min_expansion:
                        min_expansion = total_exp
                        conservative_key = (sname, val)
        if conservative_key:
            interesting.append(("conservative", conservative_key))
            used_keys.add(conservative_key)

        # 3. Best from a different sweep dimension than #1
        best_key_sweep = best_key[0] if best_key else None
        alt_best_key = None
        alt_best_recall = -1.0
        for (sname, val), res in all_results.items():
            if (sname, val) not in used_keys and not res["is_default"]:
                if sname != best_key_sweep:
                    if res["oracle_recall"] > alt_best_recall:
                        alt_best_recall = res["oracle_recall"]
                        alt_best_key = (sname, val)
        if alt_best_key:
            interesting.append(("alt_best", alt_best_key))
            used_keys.add(alt_best_key)

        for label, (sname, val) in interesting:
            res = all_results[(sname, val)]
            tag = f"{sname.split('(')[0].strip()}_{val}".replace(" ", "_")
            out_path = os.path.join(
                OUTPUT_DIR,
                f"sensitivity_{label}_{tag}_candidates.jsonl",
            )
            with open(out_path, "w") as f:
                for item in res["pool_data"]:
                    f.write(json.dumps(item) + "\n")
            print(f"  [{label}] {sname}={val}: "
                  f"oracle={res['oracle_recall']:.2f}%, "
                  f"delta={res['delta_vs_bm25']:+.2f}%")
            print(f"    -> {out_path}")

    # ---- Clean up pool_data from memory before JSON dump ----
    summary = {}
    for (sname, val), res in all_results.items():
        key_str = f"{sname}|{val}"
        summary[key_str] = {
            k: v for k, v in res.items() if k != "pool_data"
        }

    summary_path = os.path.join(
        OUTPUT_DIR, "sensitivity_expansion_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
