"""
SWE-bench Lite graph expansion pipeline.

Mines co-change edges from SWE-bench training data and loads import edges
from dep_graphs to expand BM25 candidate lists for each test example.

Outputs:
  1. data/rankft/swebench_merged_graph_candidates.jsonl
     BM25 top-500 candidates expanded with co-change + import neighbors
  2. data/rankft/swebench_bm25_only_candidates.jsonl
     BM25 top-200 baseline (for fair comparison at same candidate budget)

Output format matches eval_rankft.py expectations:
  {repo, issue_id, issue_text, ground_truth, bm25_candidates, gt_in_candidates}

Usage:
    python scripts/swebench_graph_expansion.py
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
SWEBENCH_TEST = os.path.join(BASE_DIR, "data/swebench_lite/swebench_lite_test.jsonl")
SWEBENCH_TRAIN = os.path.join(BASE_DIR, "data/swebench_train/swebench_train.jsonl")
BM25_TOP500 = os.path.join(BASE_DIR, "data/rankft/swebench_test_bm25_top500.jsonl")
DEP_GRAPH_DIR = os.path.join(BASE_DIR, "data/dep_graphs")
FILE_TREE_DIR = os.path.join(BASE_DIR, "data/swebench_file_trees")

OUTPUT_MERGED = os.path.join(BASE_DIR, "data/rankft/swebench_merged_graph_candidates.jsonl")
OUTPUT_BM25_ONLY = os.path.join(BASE_DIR, "data/rankft/swebench_bm25_only_candidates.jsonl")

# Expansion parameters (match GREPO pipeline)
MAX_COCHANGE = 10
MAX_IMPORT = 10
MIN_COCHANGE_SCORE = 0.05


# ============================================================
# Co-change index from SWE-bench training data
# ============================================================

def build_cochange_index(train_path: str, min_cochange: int = 1) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build per-repo co-change index from SWE-bench training data.

    Unlike the GREPO version in graph_expansion.py which filters on split=='train',
    SWE-bench training data has no split field -- all entries are training examples.
    """
    repo_cochanges: Dict[str, Counter] = defaultdict(Counter)
    repo_file_count: Dict[str, Counter] = defaultdict(Counter)
    total_examples = 0
    total_with_py = 0

    with open(train_path) as f:
        for line in f:
            item = json.loads(line)
            total_examples += 1

            repo = item["repo"]
            files = item.get("changed_py_files", [])
            if not files:
                files = [fp for fp in item.get("changed_files", []) if fp.endswith(".py")]

            if not files:
                continue
            total_with_py += 1

            for fp in files:
                repo_file_count[repo][fp] += 1

            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    # Build scored index
    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    total_edges = 0
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score
                total_edges += 1

    print(f"  Training examples: {total_examples} ({total_with_py} with .py files)")
    print(f"  Repos with co-change edges: {len(index)}")
    print(f"  Total co-change edges: {total_edges}")
    for repo in sorted(index.keys()):
        n_files = len(index[repo])
        n_edges = sum(len(v) for v in index[repo].values())
        print(f"    {repo}: {n_files} source files, {n_edges} edges")

    return index


# ============================================================
# Import index from dep_graphs
# ============================================================

def build_import_index(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build per-repo bidirectional import index from dep_graphs.

    Dep graph files use short repo names (e.g., 'astropy'), while SWE-bench
    uses 'astropy__astropy'. This function returns the index keyed by the
    short name; callers must map SWE-bench repo names accordingly.
    """
    index: Dict[str, Dict[str, Set[str]]] = {}

    if not os.path.isdir(dep_graph_dir):
        print(f"  WARNING: dep_graph_dir not found: {dep_graph_dir}")
        return index

    for fname in sorted(os.listdir(dep_graph_dir)):
        if not fname.endswith("_rels.json"):
            continue
        repo_short = fname.replace("_rels.json", "")

        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)

        neighbors: Dict[str, Set[str]] = defaultdict(set)

        # File-level import edges
        for src, targets in rels.get("file_imports", {}).items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)  # bidirectional

        # Call-graph based file-level edges
        for src_func, callees in rels.get("call_graph", {}).items():
            src_file = src_func.split(":")[0] if ":" in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(":")[0] if ":" in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)

        index[repo_short] = dict(neighbors)

    return index


# ============================================================
# File tree index (for validating candidates exist in repo)
# ============================================================

def build_file_tree_index(file_tree_dir: str) -> Dict[str, Set[str]]:
    """Load per-repo sets of valid .py file paths."""
    index: Dict[str, Set[str]] = {}

    if not os.path.isdir(file_tree_dir):
        print(f"  WARNING: file_tree_dir not found: {file_tree_dir}")
        return index

    for fname in sorted(os.listdir(file_tree_dir)):
        if not fname.endswith(".json"):
            continue
        repo = fname.replace(".json", "")

        with open(os.path.join(file_tree_dir, fname)) as f:
            data = json.load(f)

        py_files = set(data.get("py_files", []))
        if py_files:
            index[repo] = py_files

    return index


# ============================================================
# Repo name mapping
# ============================================================

def swebench_repo_to_dep_key(swebench_repo: str) -> str:
    """
    Map SWE-bench repo name to dep_graph key.
    e.g. 'astropy__astropy' -> 'astropy'
         'scikit-learn__scikit-learn' -> 'scikit-learn'
    """
    return swebench_repo.split("__")[-1]


# ============================================================
# Graph expansion
# ============================================================

def expand_candidates(
    bm25_candidates: List[str],
    repo: str,
    cochange_index: Dict[str, Dict[str, Dict[str, float]]],
    import_index: Dict[str, Dict[str, Set[str]]],
    file_tree: Set[str],
    max_cochange: int = MAX_COCHANGE,
    max_import: int = MAX_IMPORT,
    min_cochange_score: float = MIN_COCHANGE_SCORE,
) -> Tuple[List[str], int, int]:
    """
    Expand a BM25 candidate list with graph neighbors.

    Order: BM25 candidates -> co-change expansions -> import expansions
    Returns: (expanded_candidates, num_cochange_added, num_import_added)
    """
    original_set = set(bm25_candidates)
    dep_key = swebench_repo_to_dep_key(repo)

    # Use top BM25 candidates as seeds for expansion
    # (using top-50 as seeds is sufficient and avoids noise from low-ranked BM25 hits)
    seed_files = bm25_candidates[:50]

    # 1. Co-change expansion
    cochange_cands: Dict[str, float] = {}
    repo_cc = cochange_index.get(repo, {})
    for seed_file in seed_files:
        for neighbor, score in repo_cc.get(seed_file, {}).items():
            if neighbor not in original_set and score >= min_cochange_score:
                cochange_cands[neighbor] = max(cochange_cands.get(neighbor, 0), score)

    sorted_cc = sorted(cochange_cands.items(), key=lambda x: x[1], reverse=True)
    # Filter to valid files if we have a file tree
    if file_tree:
        cc_expansion = [c for c, _ in sorted_cc if c in file_tree][:max_cochange]
    else:
        cc_expansion = [c for c, _ in sorted_cc][:max_cochange]

    # 2. Import expansion
    expanded_so_far = original_set | set(cc_expansion)
    import_scores: Dict[str, int] = defaultdict(int)
    repo_imp = import_index.get(dep_key, {})
    for seed_file in seed_files:
        for neighbor in repo_imp.get(seed_file, set()):
            if neighbor not in expanded_so_far:
                import_scores[neighbor] += 1

    sorted_imp = sorted(import_scores.items(), key=lambda x: x[1], reverse=True)
    if file_tree:
        imp_expansion = [c for c, _ in sorted_imp if c in file_tree][:max_import]
    else:
        imp_expansion = [c for c, _ in sorted_imp][:max_import]

    # Merge: BM25 original + co-change + import
    merged = list(bm25_candidates) + cc_expansion + imp_expansion

    return merged, len(cc_expansion), len(imp_expansion)


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("=" * 70)
    print("SWE-bench Lite Graph Expansion Pipeline")
    print("=" * 70)

    # Verify inputs exist
    for path, label in [
        (SWEBENCH_TEST, "SWE-bench test data"),
        (SWEBENCH_TRAIN, "SWE-bench training data"),
        (BM25_TOP500, "BM25 top-500 candidates"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # --- Step 1: Build co-change index ---
    print("\n[1/4] Building co-change index from SWE-bench training data...")
    cc_index = build_cochange_index(SWEBENCH_TRAIN, min_cochange=1)

    # --- Step 2: Build import index ---
    print("\n[2/4] Building import index from dep_graphs...")
    imp_index = build_import_index(DEP_GRAPH_DIR)

    # Map SWE-bench repo names to dep_graph keys for coverage report
    swe_repos = set()
    with open(SWEBENCH_TEST) as f:
        for line in f:
            swe_repos.add(json.loads(line)["repo"])

    print(f"  Import index has {len(imp_index)} repos")
    print(f"  SWE-bench has {len(swe_repos)} unique repos")
    covered = 0
    for sr in sorted(swe_repos):
        dk = swebench_repo_to_dep_key(sr)
        has_dep = dk in imp_index
        has_cc = sr in cc_index
        if has_dep:
            n_files = len(imp_index[dk])
            n_edges = sum(len(v) for v in imp_index[dk].values())
            covered += 1
        else:
            n_files = 0
            n_edges = 0
        print(f"    {sr}: dep_graph={'YES' if has_dep else 'NO'} "
              f"({n_files} files, {n_edges} edges), "
              f"co-change={'YES' if has_cc else 'NO'}")
    print(f"  Import coverage: {covered}/{len(swe_repos)} repos")

    # --- Step 3: Build file tree index ---
    print("\n[3/4] Loading file trees for candidate validation...")
    ft_index = build_file_tree_index(FILE_TREE_DIR)
    print(f"  File trees loaded for {len(ft_index)} repos")
    for repo, files in sorted(ft_index.items()):
        print(f"    {repo}: {len(files)} .py files")

    # --- Step 4: Process each test example ---
    print(f"\n[4/4] Expanding BM25 candidates with graph neighbors...")
    print(f"  Parameters: max_cochange={MAX_COCHANGE}, max_import={MAX_IMPORT}, "
          f"min_cochange_score={MIN_COCHANGE_SCORE}")

    # Load BM25 data
    bm25_data = []
    with open(BM25_TOP500) as f:
        for line in f:
            bm25_data.append(json.loads(line))
    print(f"  Loaded {len(bm25_data)} BM25 examples")

    # Process
    merged_results = []
    bm25_only_results = []

    stats = {
        "total": len(bm25_data),
        "cc_expanded": 0,
        "imp_expanded": 0,
        "total_cc_added": 0,
        "total_imp_added": 0,
        "gt_in_bm25_500": 0,
        "gt_in_bm25_200": 0,
        "gt_in_merged": 0,
        "gt_new_from_cc": 0,
        "gt_new_from_imp": 0,
    }

    for item in bm25_data:
        repo = item["repo"]
        issue_id = item["issue_id"]
        issue_text = item["issue_text"]
        ground_truth = item["ground_truth"]
        bm25_cands = item["bm25_candidates"]
        gt_set = set(ground_truth)
        ft = ft_index.get(repo, set())

        # Expand
        merged_cands, n_cc, n_imp = expand_candidates(
            bm25_cands, repo, cc_index, imp_index, ft,
            max_cochange=MAX_COCHANGE,
            max_import=MAX_IMPORT,
            min_cochange_score=MIN_COCHANGE_SCORE,
        )

        # Stats
        if n_cc > 0:
            stats["cc_expanded"] += 1
        if n_imp > 0:
            stats["imp_expanded"] += 1
        stats["total_cc_added"] += n_cc
        stats["total_imp_added"] += n_imp

        bm25_set_500 = set(bm25_cands[:500])
        bm25_set_200 = set(bm25_cands[:200])
        merged_set = set(merged_cands)

        if gt_set & bm25_set_500:
            stats["gt_in_bm25_500"] += 1
        if gt_set & bm25_set_200:
            stats["gt_in_bm25_200"] += 1
        if gt_set & merged_set:
            stats["gt_in_merged"] += 1

        # Check if graph expansion added GT files not in original BM25
        cc_added = set(merged_cands[len(bm25_cands):len(bm25_cands) + n_cc])
        imp_added = set(merged_cands[len(bm25_cands) + n_cc:])
        if gt_set & cc_added:
            stats["gt_new_from_cc"] += 1
        if gt_set & imp_added:
            stats["gt_new_from_imp"] += 1

        # Output 1: Merged graph candidates
        merged_entry = {
            "repo": repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "ground_truth": ground_truth,
            "bm25_candidates": merged_cands,
            "gt_in_candidates": bool(gt_set & merged_set),
            "num_bm25_original": len(bm25_cands),
            "num_cochange_added": n_cc,
            "num_import_added": n_imp,
            "num_total_candidates": len(merged_cands),
        }
        merged_results.append(merged_entry)

        # Output 2: BM25-only top-200
        bm25_200 = bm25_cands[:200]
        bm25_only_entry = {
            "repo": repo,
            "issue_id": issue_id,
            "issue_text": issue_text,
            "ground_truth": ground_truth,
            "bm25_candidates": bm25_200,
            "gt_in_candidates": bool(gt_set & set(bm25_200)),
        }
        bm25_only_results.append(bm25_only_entry)

    # Write outputs
    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)

    with open(OUTPUT_MERGED, "w") as f:
        for entry in merged_results:
            f.write(json.dumps(entry) + "\n")
    print(f"\n  Written: {OUTPUT_MERGED}")
    print(f"    {len(merged_results)} examples")

    with open(OUTPUT_BM25_ONLY, "w") as f:
        for entry in bm25_only_results:
            f.write(json.dumps(entry) + "\n")
    print(f"  Written: {OUTPUT_BM25_ONLY}")
    print(f"    {len(bm25_only_results)} examples")

    # --- Print stats ---
    print(f"\n{'=' * 70}")
    print("EXPANSION STATISTICS")
    print(f"{'=' * 70}")
    n = stats["total"]
    print(f"  Total examples: {n}")
    print(f"  Examples with co-change expansion: {stats['cc_expanded']}/{n} "
          f"({100*stats['cc_expanded']/n:.1f}%)")
    print(f"  Examples with import expansion: {stats['imp_expanded']}/{n} "
          f"({100*stats['imp_expanded']/n:.1f}%)")
    avg_cc = stats["total_cc_added"] / n
    avg_imp = stats["total_imp_added"] / n
    print(f"  Avg co-change neighbors added: {avg_cc:.2f}")
    print(f"  Avg import neighbors added: {avg_imp:.2f}")
    avg_total = sum(e["num_total_candidates"] for e in merged_results) / n
    print(f"  Avg total candidates (merged): {avg_total:.1f}")

    print(f"\nGROUND TRUTH COVERAGE")
    print(f"  GT in BM25-top-500: {stats['gt_in_bm25_500']}/{n} "
          f"({100*stats['gt_in_bm25_500']/n:.1f}%)")
    print(f"  GT in BM25-top-200: {stats['gt_in_bm25_200']}/{n} "
          f"({100*stats['gt_in_bm25_200']/n:.1f}%)")
    print(f"  GT in merged (BM25+graph): {stats['gt_in_merged']}/{n} "
          f"({100*stats['gt_in_merged']/n:.1f}%)")
    new_from_graph = stats["gt_in_merged"] - stats["gt_in_bm25_500"]
    print(f"  NEW GT found by graph expansion: {new_from_graph}")
    print(f"    via co-change: {stats['gt_new_from_cc']}")
    print(f"    via import: {stats['gt_new_from_imp']}")

    # Per-repo breakdown
    print(f"\nPER-REPO BREAKDOWN")
    repo_stats = defaultdict(lambda: {
        "count": 0, "cc_added": 0, "imp_added": 0,
        "gt_in_bm25_200": 0, "gt_in_bm25_500": 0, "gt_in_merged": 0,
        "gt_new_graph": 0,
    })
    for entry in merged_results:
        repo = entry["repo"]
        gt_set = set(entry["ground_truth"])
        n_orig = entry["num_bm25_original"]
        bm25_orig = entry["bm25_candidates"][:n_orig]
        graph_added = entry["bm25_candidates"][n_orig:]
        repo_stats[repo]["count"] += 1
        repo_stats[repo]["cc_added"] += entry["num_cochange_added"]
        repo_stats[repo]["imp_added"] += entry["num_import_added"]
        if gt_set & set(bm25_orig[:200]):
            repo_stats[repo]["gt_in_bm25_200"] += 1
        if gt_set & set(bm25_orig):
            repo_stats[repo]["gt_in_bm25_500"] += 1
        if entry["gt_in_candidates"]:
            repo_stats[repo]["gt_in_merged"] += 1
        if gt_set & set(graph_added):
            repo_stats[repo]["gt_new_graph"] += 1

    header = (f"{'Repo':<28} {'N':>4} {'AvgCC':>6} {'AvgImp':>6} "
              f"{'GT@200':>7} {'GT@500':>7} {'GT@Mrg':>7} {'Graph+':>6}")
    print(header)
    print("-" * len(header))
    for repo in sorted(repo_stats.keys()):
        rs = repo_stats[repo]
        cnt = rs["count"]
        avg_cc_r = rs["cc_added"] / cnt
        avg_imp_r = rs["imp_added"] / cnt
        gt200 = rs["gt_in_bm25_200"]
        gt500 = rs["gt_in_bm25_500"]
        gtm = rs["gt_in_merged"]
        graph_new = rs["gt_new_graph"]
        print(f"  {repo:<26} {cnt:>4} {avg_cc_r:>6.2f} {avg_imp_r:>6.2f} "
              f"{gt200:>4}/{cnt:<3} {gt500:>4}/{cnt:<3} {gtm:>4}/{cnt:<3} {graph_new:>+4}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
