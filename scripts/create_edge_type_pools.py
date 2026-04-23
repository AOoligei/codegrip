#!/usr/bin/env python3
"""
Create edge-type ablation candidate pools for Stage 2 graph expansion.

Generates three candidate pools:
1. cochange_only  : BM25 + co-change neighbors (no import edges)
2. import_only    : BM25 + import neighbors (no co-change edges)
3. both           : BM25 + co-change + import (reproduces current merged pool)

Each pool follows the same construction as merged_bm25_exp6_candidates.jsonl:
  - Graph neighbors placed first (preserving score-based ordering)
  - BM25 candidates fill the rest (deduped)
  - Truncated to BM25 pool size

Outputs saved to data/rankft/ as JSONL files.

Usage:
    python scripts/create_edge_type_pools.py
    python scripts/create_edge_type_pools.py --dry_run   # stats only
"""

import json
import os
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set

random.seed(42)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BM25_CANDIDATES = os.path.join(
    PROJECT_ROOT, "data", "rankft", "bm25_top_matched_candidates.jsonl"
)
TRAIN_DATA = os.path.join(
    PROJECT_ROOT, "data", "grepo_text", "grepo_train.jsonl"
)
DEP_GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "dep_graphs")
TEST_DATA = os.path.join(
    PROJECT_ROOT, "data", "grepo_text", "grepo_test.jsonl"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "rankft")


# ============================================================
# Co-change index (from training PR data)
# ============================================================

def build_cochange_index(
    train_data_path: str, min_cochange: int = 1
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build per-repo co-change index from training PR data.
    Returns {repo: {file_a: {file_b: score}}}
    where score = co_change_count(a, b) / total_changes(a).
    """
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


# ============================================================
# Import index (from dep_graphs)
# ============================================================

def build_import_index(
    dep_graph_dir: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build per-repo bidirectional import index.
    Includes file_imports and call_graph edges (file-level).
    Returns {repo: {file: set of neighbor files}}.
    """
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

        # file_imports: source -> [targets]
        for src, targets in rels.get("file_imports", {}).items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)  # bidirectional

        # call_graph: func_or_file -> [callees]
        for src_func, callees in rels.get("call_graph", {}).items():
            src_file = src_func.split(":")[0] if ":" in src_func else src_func
            for callee in callees:
                tgt_file = (
                    callee.split(":")[0] if ":" in callee else callee
                )
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)

        index[repo] = dict(neighbors)

    return index


# ============================================================
# Expansion logic
# ============================================================

def expand_with_cochange(
    seed_files: List[str],
    cochange_index: Dict[str, Dict[str, float]],
    max_expand: int = 10,
    min_score: float = 0.05,
) -> List[str]:
    """Get co-change neighbors for a set of seed files, sorted by score."""
    seed_set = set(seed_files)
    cands: Dict[str, float] = {}

    for pred_file in seed_files:
        for neighbor, score in cochange_index.get(pred_file, {}).items():
            if neighbor not in seed_set and score >= min_score:
                cands[neighbor] = max(cands.get(neighbor, 0), score)

    sorted_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_cands[:max_expand]]


def expand_with_imports(
    seed_files: List[str],
    import_index: Dict[str, Set[str]],
    exclude: Set[str],
    max_expand: int = 10,
) -> List[str]:
    """Get import neighbors for a set of seed files, scored by link count."""
    scores: Dict[str, int] = defaultdict(int)

    for pred_file in seed_files:
        for neighbor in import_index.get(pred_file, set()):
            if neighbor not in exclude:
                scores[neighbor] += 1

    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_cands[:max_expand]]


def build_expansion_candidates(
    bm25_candidates: List[str],
    repo: str,
    cochange_index: Dict[str, Dict[str, Dict[str, float]]],
    import_index: Dict[str, Dict[str, Set[str]]],
    mode: str = "both",
    max_cochange: int = 10,
    max_import: int = 10,
    min_cochange_score: float = 0.05,
) -> List[str]:
    """
    Build expanded candidate pool for a single example.

    Modes:
        "cochange_only" : BM25 seed -> co-change neighbors only
        "import_only"   : BM25 seed -> import neighbors only
        "both"          : BM25 seed -> co-change then import neighbors

    Construction follows merged_bm25_exp6 convention:
        graph_neighbors (ordered by score) placed first,
        then BM25 candidates fill the rest (deduped),
        truncated to BM25 pool size.
    """
    bm25_set = set(bm25_candidates)

    # Use top BM25 candidates as seed for graph expansion
    # (matching how graph_expansion.py uses "original predictions")
    seed = bm25_candidates[:20]

    cc_expansion = []
    imp_expansion = []

    repo_cc = cochange_index.get(repo, {})
    repo_imp = import_index.get(repo, {})

    if mode in ("cochange_only", "both"):
        cc_expansion = expand_with_cochange(
            seed, repo_cc,
            max_expand=max_cochange,
            min_score=min_cochange_score,
        )

    if mode in ("import_only", "both"):
        already = set(seed) | set(cc_expansion)
        imp_expansion = expand_with_imports(
            seed, repo_imp,
            exclude=already,
            max_expand=max_import,
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

    # Truncate to BM25 pool size (same convention as original merge)
    merged = merged[: len(bm25_candidates)]

    return merged


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create edge-type ablation candidate pools"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print statistics only, do not write files",
    )
    parser.add_argument(
        "--max_cochange", type=int, default=10,
        help="Max co-change neighbors per example",
    )
    parser.add_argument(
        "--max_import", type=int, default=10,
        help="Max import neighbors per example",
    )
    parser.add_argument(
        "--min_cochange_score", type=float, default=0.05,
        help="Min co-change score threshold",
    )
    args = parser.parse_args()

    # ---- Load BM25 candidate pools ----
    print("Loading BM25 candidates...")
    bm25_data = {}
    with open(BM25_CANDIDATES) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            bm25_data[key] = d["candidates"]
    print(f"  {len(bm25_data)} examples")

    # ---- Build co-change index ----
    print("Building co-change index from training data...")
    cc_index = build_cochange_index(TRAIN_DATA, min_cochange=1)
    cc_repos = len(cc_index)
    cc_pairs = sum(
        len(neighbors)
        for repo_idx in cc_index.values()
        for neighbors in repo_idx.values()
    )
    print(f"  {cc_repos} repos, {cc_pairs} directed co-change pairs")

    # ---- Build import index ----
    print("Building import index from dep_graphs...")
    imp_index = build_import_index(DEP_GRAPH_DIR)
    imp_repos = len(imp_index)
    imp_edges = sum(
        len(neighbors)
        for repo_idx in imp_index.values()
        for neighbors in repo_idx.values()
    )
    print(f"  {imp_repos} repos, {imp_edges} bidirectional import edges")

    # ---- Generate pools for each mode ----
    modes = {
        "cochange_only": "merged_bm25_cochange_only_candidates.jsonl",
        "import_only": "merged_bm25_import_only_candidates.jsonl",
        "both": "merged_bm25_both_edge_types_candidates.jsonl",
    }

    for mode, output_fname in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        output_path = os.path.join(OUTPUT_DIR, output_fname)
        results = []
        n_expanded = 0
        total_graph_additions = 0

        for key in sorted(bm25_data.keys()):
            repo, issue_id = key
            bm25_cands = bm25_data[key]

            merged = build_expansion_candidates(
                bm25_cands,
                repo,
                cc_index,
                imp_index,
                mode=mode,
                max_cochange=args.max_cochange,
                max_import=args.max_import,
                min_cochange_score=args.min_cochange_score,
            )

            # Count how many graph neighbors were added
            bm25_set = set(bm25_cands)
            n_graph = sum(1 for c in merged if c not in bm25_set)
            if n_graph > 0:
                n_expanded += 1
            total_graph_additions += n_graph

            results.append({
                "repo": repo,
                "issue_id": issue_id,
                "candidates": merged,
            })

        # Stats
        pool_sizes = [len(r["candidates"]) for r in results]
        avg_pool = sum(pool_sizes) / len(pool_sizes)
        avg_graph = total_graph_additions / len(results)
        print(f"  Examples: {len(results)}")
        print(f"  Expanded (>0 graph neighbors): {n_expanded}/{len(results)} "
              f"({n_expanded/len(results)*100:.1f}%)")
        print(f"  Avg graph additions: {avg_graph:.1f}")
        print(f"  Avg pool size: {avg_pool:.1f}")

        if not args.dry_run:
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved to: {output_path}")
        else:
            print(f"  [dry_run] Would save to: {output_path}")

    # ---- Comparison with existing merged pool ----
    existing_merged_path = os.path.join(
        OUTPUT_DIR, "merged_bm25_exp6_candidates.jsonl"
    )
    if os.path.exists(existing_merged_path):
        print(f"\n{'='*60}")
        print("Comparison with existing merged_bm25_exp6_candidates.jsonl")
        print(f"{'='*60}")
        with open(existing_merged_path) as f:
            existing = {
                (json.loads(l)["repo"], json.loads(l)["issue_id"]):
                json.loads(l)["candidates"]
                for l in f
            }

        # Load the "both" pool we just created (or would create)
        both_path = os.path.join(
            OUTPUT_DIR, "merged_bm25_both_edge_types_candidates.jsonl"
        )
        if os.path.exists(both_path):
            with open(both_path) as f:
                both_pool = {
                    (json.loads(l)["repo"], json.loads(l)["issue_id"]):
                    json.loads(l)["candidates"]
                    for l in f
                }
            overlap_count = 0
            total_overlap_jaccard = 0.0
            for key in existing:
                if key in both_pool:
                    e_set = set(existing[key])
                    b_set = set(both_pool[key])
                    jaccard = len(e_set & b_set) / max(len(e_set | b_set), 1)
                    total_overlap_jaccard += jaccard
                    overlap_count += 1
            if overlap_count > 0:
                avg_jaccard = total_overlap_jaccard / overlap_count
                print(f"  Avg Jaccard similarity (both vs existing): "
                      f"{avg_jaccard:.3f} over {overlap_count} examples")


if __name__ == "__main__":
    main()
