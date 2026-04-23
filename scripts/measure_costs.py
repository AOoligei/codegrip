#!/usr/bin/env python3
"""Measure wall-clock costs for the CodeGRIP pipeline.

Profiles each stage of the pipeline:
  1. Graph construction: co-change index + import index building
  2. Graph expansion: per-example candidate expansion (1704 test examples)
  3. Candidate pool sizes: storage footprint of key data files
  4. Cross-encoder reranking: estimated from eval logs or measured

Usage:
    python scripts/measure_costs.py
    python scripts/measure_costs.py --runs 3          # average over 3 runs
    python scripts/measure_costs.py --skip_rerank     # skip reranking estimate
"""

import argparse
import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Paths
# ============================================================
TRAIN_DATA = os.path.join(PROJECT_ROOT, "data", "grepo_text", "grepo_train.jsonl")
TEST_DATA = os.path.join(PROJECT_ROOT, "data", "grepo_text", "grepo_test.jsonl")
DEP_GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "dep_graphs")
FILE_TREE_DIR = os.path.join(PROJECT_ROOT, "data", "file_trees")

BM25_CANDIDATES = os.path.join(
    PROJECT_ROOT, "data", "rankft", "bm25_top_matched_candidates.jsonl"
)
EXPANDED_CANDIDATES = os.path.join(
    PROJECT_ROOT, "data", "rankft", "merged_bm25_both_edge_types_candidates.jsonl"
)
BM25_TOP500 = os.path.join(
    PROJECT_ROOT, "data", "rankft", "grepo_test_bm25_top500.jsonl"
)

# Reranker eval log with timing info
RERANK_LOG = os.path.join(
    PROJECT_ROOT, "experiments", "rankft_runB_graph",
    "eval_merged_rerank.log"
)


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.2f}h"


def format_size(path):
    """Format file size in human-readable form."""
    if not os.path.exists(path):
        return "-"
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size}B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.0f}KB"
    else:
        return f"{size / 1024 ** 2:.0f}MB"


def dir_size(path):
    """Total size of all files in a directory."""
    total = 0
    if os.path.isdir(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


# ============================================================
# Stage 1: Graph Construction
# ============================================================

def measure_graph_construction(n_runs=1):
    """Measure co-change index and import index build time."""
    from src.eval.multi_signal_expansion import build_cochange_index, build_import_index

    # --- Co-change index ---
    cc_times = []
    cc_stats = {}
    for _ in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        cc_index = build_cochange_index(TRAIN_DATA, min_cochange=1)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cc_times.append(elapsed)

    n_repos_cc = len(cc_index)
    total_pairs = sum(
        sum(len(v) for v in repo.values())
        for repo in cc_index.values()
    )
    cc_stats = {
        "time_sec": sum(cc_times) / len(cc_times),
        "n_repos": n_repos_cc,
        "total_pairs": total_pairs,
        "peak_mb": peak / 1024 / 1024,
    }

    # --- Import index ---
    imp_times = []
    imp_stats = {}
    for _ in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        imp_index = build_import_index(DEP_GRAPH_DIR)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        imp_times.append(elapsed)

    n_repos_imp = len(imp_index)
    total_edges = sum(
        sum(len(v) for v in repo.values())
        for repo in imp_index.values()
    )
    imp_stats = {
        "time_sec": sum(imp_times) / len(imp_times),
        "n_repos": n_repos_imp,
        "total_edges": total_edges,
        "peak_mb": peak / 1024 / 1024,
    }

    # Count train PRs
    n_train = 0
    with open(TRAIN_DATA) as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") == "train":
                n_train += 1

    # Count dep_graph rels files
    n_rels = len([f for f in os.listdir(DEP_GRAPH_DIR) if f.endswith("_rels.json")])

    return cc_stats, imp_stats, n_train, n_rels


# ============================================================
# Stage 2: Graph Expansion
# ============================================================

def measure_graph_expansion(n_runs=1):
    """Measure graph expansion time over all test examples."""
    from src.eval.multi_signal_expansion import (
        build_cochange_index,
        build_import_index,
        build_dir_index,
    )

    # Build indices (not timed -- measured separately above)
    cc_index = build_cochange_index(TRAIN_DATA, min_cochange=1)
    imp_index = build_import_index(DEP_GRAPH_DIR)
    dir_index, all_py = build_dir_index(FILE_TREE_DIR)

    # Load BM25 candidates to simulate expansion from
    bm25_data = load_jsonl(BM25_CANDIDATES)
    bm25_lookup = {}
    for d in bm25_data:
        key = (d["repo"], d["issue_id"])
        bm25_lookup[key] = d["candidates"]

    # Load test data for ground truth
    test_data = load_jsonl(TEST_DATA)
    test_lookup = {}
    for d in test_data:
        key = (d["repo"], d["issue_id"])
        test_lookup[key] = d

    # Build a mock predictions file matching how expand_predictions expects input
    # We need: repo, predicted, ground_truth
    # Use the first ~10 BM25 candidates as "base predictions" to expand from
    mock_preds = []
    for key, cands in bm25_lookup.items():
        if key not in test_lookup:
            continue
        td = test_lookup[key]
        mock_preds.append({
            "repo": td["repo"],
            "issue_id": td["issue_id"],
            "predicted": cands[:10],  # simulate base preds
            "ground_truth": td.get("changed_py_files", []),
        })

    n_examples = len(mock_preds)

    # Time the expansion loop (inline, without file I/O)
    expansion_times = []
    total_expanded = 0

    for _ in range(n_runs):
        t0 = time.perf_counter()

        for p in mock_preds:
            repo = p["repo"]
            original = list(p["predicted"])
            original_set = set(original)
            repo_files = all_py.get(repo, set())
            scores = defaultdict(float)
            signal_count = defaultdict(int)

            # Co-change
            repo_cc = cc_index.get(repo, {})
            for pred_file in original:
                for neighbor, score in repo_cc.get(pred_file, {}).items():
                    if neighbor not in original_set and score >= 0.02:
                        scores[neighbor] += 1.0 * score
                        signal_count[neighbor] += 1

            # Import
            repo_imp = imp_index.get(repo, {})
            for pred_file in original:
                for neighbor in repo_imp.get(pred_file, set()):
                    if neighbor not in original_set:
                        scores[neighbor] += 0.6
                        signal_count[neighbor] += 1

            # Directory proximity
            repo_dir = dir_index.get(repo, {})
            pred_dirs = set(os.path.dirname(f) for f in original)
            for d in pred_dirs:
                dir_files = repo_dir.get(d, [])
                if len(dir_files) > 35:
                    continue
                specificity = 1.0 / max(len(dir_files), 1)
                for f in dir_files:
                    if f not in original_set:
                        scores[f] += 0.25 * specificity
                        signal_count[f] += 1

            # Test-source matching
            for pred_file in original:
                basename = os.path.basename(pred_file)
                dirname = os.path.dirname(pred_file)
                pairs = []
                if basename.startswith("test_"):
                    source = basename[5:]
                    parent = os.path.dirname(dirname)
                    pairs.extend([
                        os.path.join(dirname, source),
                        os.path.join(parent, source),
                    ])
                else:
                    pairs.append(os.path.join(dirname, "test_" + basename))
                    pairs.append(os.path.join(dirname, "tests", "test_" + basename))
                for pair in pairs:
                    if pair in repo_files and pair not in original_set:
                        scores[pair] += 0.7
                        signal_count[pair] += 1

            # Multi-signal boost
            for f in list(scores.keys()):
                if signal_count[f] >= 2:
                    scores[f] *= 1.3
                if signal_count[f] >= 3:
                    scores[f] *= 1.2

            sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            expansion = [c[0] for c in sorted_cands[:35]]
            total_expanded += len(expansion)

        elapsed = time.perf_counter() - t0
        expansion_times.append(elapsed)

    avg_time = sum(expansion_times) / len(expansion_times)
    avg_expanded = total_expanded / (n_examples * n_runs) if n_examples > 0 else 0

    return {
        "total_sec": avg_time,
        "n_examples": n_examples,
        "per_example_ms": (avg_time / max(n_examples, 1)) * 1000,
        "avg_expanded_neighbors": avg_expanded,
    }


# ============================================================
# Stage 3: Candidate Pool Sizes
# ============================================================

def measure_pool_sizes():
    """Report storage sizes and statistics for key data files."""
    pools = {}

    # BM25 top-500
    if os.path.exists(BM25_TOP500):
        data = load_jsonl(BM25_TOP500)
        key = "bm25_candidates" if "bm25_candidates" in data[0] else "candidates"
        lens = [len(d.get(key, [])) for d in data]
        pools["BM25 top-500 (all test)"] = {
            "path": BM25_TOP500,
            "size": format_size(BM25_TOP500),
            "n_examples": len(data),
            "avg_candidates": sum(lens) / len(lens) if lens else 0,
        }

    # BM25 matched (test only, 1704)
    if os.path.exists(BM25_CANDIDATES):
        data = load_jsonl(BM25_CANDIDATES)
        lens = [len(d.get("candidates", [])) for d in data]
        pools["BM25 matched pool"] = {
            "path": BM25_CANDIDATES,
            "size": format_size(BM25_CANDIDATES),
            "n_examples": len(data),
            "avg_candidates": sum(lens) / len(lens) if lens else 0,
        }

    # Expanded (BM25 + graph)
    if os.path.exists(EXPANDED_CANDIDATES):
        data = load_jsonl(EXPANDED_CANDIDATES)
        lens = [len(d.get("candidates", [])) for d in data]
        pools["BM25+Graph expanded"] = {
            "path": EXPANDED_CANDIDATES,
            "size": format_size(EXPANDED_CANDIDATES),
            "n_examples": len(data),
            "avg_candidates": sum(lens) / len(lens) if lens else 0,
        }

    # dep_graphs directory (rels files only, used for import index)
    rels_files = [f for f in os.listdir(DEP_GRAPH_DIR) if f.endswith("_rels.json")]
    rels_total = sum(
        os.path.getsize(os.path.join(DEP_GRAPH_DIR, f)) for f in rels_files
    )
    pools["Dependency graphs (*_rels)"] = {
        "path": DEP_GRAPH_DIR,
        "size": f"{rels_total / 1024 / 1024:.0f}MB",
        "n_examples": len(rels_files),
        "avg_candidates": 0,
    }

    return pools


# ============================================================
# Stage 4: Cross-Encoder Reranking Time
# ============================================================

def estimate_reranking():
    """Extract reranking time from eval logs, or estimate."""
    stats = {
        "total_sec": None,
        "per_example_sec": None,
        "n_examples": 1704,
        "avg_candidates": 208.5,
        "source": None,
    }

    # Try parsing the eval log
    if os.path.exists(RERANK_LOG):
        with open(RERANK_LOG) as f:
            for line in f:
                line = line.strip()
                if "Total time:" in line:
                    # "  Total time: 9192s (2.55h)"
                    parts = line.split("Total time:")[1].strip()
                    secs = parts.split("s")[0].strip()
                    try:
                        stats["total_sec"] = float(secs)
                        stats["source"] = f"parsed from {RERANK_LOG}"
                    except ValueError:
                        pass
                elif "Avg time/example:" in line:
                    parts = line.split("Avg time/example:")[1].strip()
                    secs = parts.split("s")[0].strip()
                    try:
                        stats["per_example_sec"] = float(secs)
                    except ValueError:
                        pass

    # Fallback: estimate from batch size and model size
    if stats["total_sec"] is None:
        # Qwen2.5-7B, batch=16, ~200 candidates/example, ~5.4s/example
        est_per_ex = 5.4
        stats["total_sec"] = est_per_ex * stats["n_examples"]
        stats["per_example_sec"] = est_per_ex
        stats["source"] = "estimated (Qwen2.5-7B, batch=16, 1x RTX 4090)"

    if stats["per_example_sec"] is None and stats["total_sec"] is not None:
        stats["per_example_sec"] = stats["total_sec"] / stats["n_examples"]

    return stats


# ============================================================
# Output
# ============================================================

def print_results(cc_stats, imp_stats, n_train, n_rels,
                  expansion_stats, pool_sizes, rerank_stats):
    """Print a clean summary table."""
    sep = "-" * 90
    header = f"{'Stage':<30} | {'Time':>12} | {'Storage':>10} | Notes"
    print()
    print("=" * 90)
    print("  CodeGRIP Pipeline Cost Measurement")
    print("=" * 90)
    print()
    print(header)
    print(sep)

    # Co-change index
    print(
        f"{'Co-change index build':<30} | "
        f"{format_time(cc_stats['time_sec']):>12} | "
        f"{'—':>10} | "
        f"{n_train:,} train PRs, {cc_stats['n_repos']} repos, "
        f"{cc_stats['total_pairs']:,} pairs"
    )

    # Import index
    rels_size = sum(
        os.path.getsize(os.path.join(DEP_GRAPH_DIR, f))
        for f in os.listdir(DEP_GRAPH_DIR) if f.endswith("_rels.json")
    ) if os.path.isdir(DEP_GRAPH_DIR) else 0
    rels_size_str = f"{rels_size / 1024 / 1024:.0f}MB" if rels_size > 0 else "—"
    print(
        f"{'Import index load':<30} | "
        f"{format_time(imp_stats['time_sec']):>12} | "
        f"{rels_size_str:>10} | "
        f"{n_rels} repos, {imp_stats['total_edges']:,} edges"
    )

    # Graph expansion
    exp = expansion_stats
    exp_storage = format_size(EXPANDED_CANDIDATES) if os.path.exists(EXPANDED_CANDIDATES) else "—"
    print(
        f"{'Graph expansion (' + str(exp['n_examples']) + ')' :<30} | "
        f"{format_time(exp['total_sec']):>12} | "
        f"{exp_storage:>10} | "
        f"avg {exp['avg_expanded_neighbors']:.1f} neighbors/example, "
        f"{exp['per_example_ms']:.2f}ms/ex"
    )

    # Reranking
    rr = rerank_stats
    rr_time = format_time(rr["total_sec"]) if rr["total_sec"] else "—"
    print(
        f"{'Cross-encoder reranking':<30} | "
        f"{'~' + rr_time:>12} | "
        f"{'—':>10} | "
        f"{rr['n_examples']} x {rr['avg_candidates']:.0f} pairs, 7B model"
    )

    print(sep)

    # Total
    graph_total = cc_stats["time_sec"] + imp_stats["time_sec"] + exp["total_sec"]
    pipeline_total = graph_total + (rr["total_sec"] or 0)
    print(
        f"{'Graph overhead (stages 1-3)':<30} | "
        f"{format_time(graph_total):>12} | "
        f"{'—':>10} | "
        f"CPU only, negligible vs reranking"
    )
    print(
        f"{'Total pipeline':<30} | "
        f"{'~' + format_time(pipeline_total):>12} | "
        f"{'—':>10} | "
        f"single GPU (RTX 4090)"
    )

    print()
    print("=" * 90)
    print("  Candidate Pool Sizes")
    print("=" * 90)
    print()
    pool_header = f"{'Pool':<30} | {'Size':>10} | {'Examples':>10} | {'Avg Cands':>10}"
    print(pool_header)
    print("-" * 70)
    for name, info in pool_sizes.items():
        avg = f"{info['avg_candidates']:.1f}" if info["avg_candidates"] > 0 else "—"
        print(
            f"{name:<30} | "
            f"{info['size']:>10} | "
            f"{info['n_examples']:>10} | "
            f"{avg:>10}"
        )

    print()
    print("=" * 90)
    print("  Overhead Analysis")
    print("=" * 90)
    print()
    if rr["total_sec"]:
        overhead_pct = (graph_total / rr["total_sec"]) * 100
        print(f"  Graph construction + expansion: {format_time(graph_total)}")
        print(f"  Cross-encoder reranking:        ~{format_time(rr['total_sec'])}")
        print(f"  Graph overhead as % of reranking: {overhead_pct:.1f}%")
        print(f"  Graph overhead as % of total:     {(graph_total / pipeline_total) * 100:.1f}%")
    print()


def save_json_stats(cc_stats, imp_stats, n_train, n_rels,
                    expansion_stats, pool_sizes, rerank_stats, output_path):
    """Save raw measurements as JSON."""
    stats = {
        "cochange_index": {
            "time_sec": cc_stats["time_sec"],
            "n_repos": cc_stats["n_repos"],
            "total_pairs": cc_stats["total_pairs"],
            "peak_mb": cc_stats["peak_mb"],
            "n_train_prs": n_train,
        },
        "import_index": {
            "time_sec": imp_stats["time_sec"],
            "n_repos": imp_stats["n_repos"],
            "total_edges": imp_stats["total_edges"],
            "peak_mb": imp_stats["peak_mb"],
            "n_rels_files": n_rels,
        },
        "graph_expansion": expansion_stats,
        "cross_encoder_reranking": rerank_stats,
        "pool_sizes": {
            name: {k: v for k, v in info.items() if k != "path"}
            for name, info in pool_sizes.items()
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Raw stats saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure wall-clock costs for the CodeGRIP pipeline"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs to average timing over (default: 1)"
    )
    parser.add_argument(
        "--skip_rerank", action="store_true",
        help="Skip reranking time estimation"
    )
    parser.add_argument(
        "--output", default=os.path.join(PROJECT_ROOT, "docs", "tables", "pipeline_costs.json"),
        help="Path to save JSON stats"
    )
    args = parser.parse_args()

    print(f"Measuring CodeGRIP pipeline costs (averaging over {args.runs} run(s))...\n")

    # Verify data files exist
    for path, label in [
        (TRAIN_DATA, "Training data"),
        (DEP_GRAPH_DIR, "Dep graphs dir"),
        (FILE_TREE_DIR, "File trees dir"),
        (BM25_CANDIDATES, "BM25 candidates"),
    ]:
        if not os.path.exists(path):
            print(f"WARNING: {label} not found at {path}")

    # Stage 1: Graph construction
    print("--- Stage 1: Graph Construction ---")
    cc_stats, imp_stats, n_train, n_rels = measure_graph_construction(n_runs=args.runs)
    print(f"  Co-change index: {format_time(cc_stats['time_sec'])} "
          f"({cc_stats['n_repos']} repos, {cc_stats['total_pairs']:,} pairs, "
          f"{cc_stats['peak_mb']:.0f}MB peak)")
    print(f"  Import index:    {format_time(imp_stats['time_sec'])} "
          f"({imp_stats['n_repos']} repos, {imp_stats['total_edges']:,} edges, "
          f"{imp_stats['peak_mb']:.0f}MB peak)")

    # Stage 2: Graph expansion
    print("\n--- Stage 2: Graph Expansion ---")
    expansion_stats = measure_graph_expansion(n_runs=args.runs)
    print(f"  {expansion_stats['n_examples']} examples in "
          f"{format_time(expansion_stats['total_sec'])} "
          f"({expansion_stats['per_example_ms']:.2f}ms/example)")
    print(f"  Avg {expansion_stats['avg_expanded_neighbors']:.1f} neighbors/example")

    # Stage 3: Pool sizes
    print("\n--- Stage 3: Candidate Pool Sizes ---")
    pool_sizes = measure_pool_sizes()
    for name, info in pool_sizes.items():
        print(f"  {name}: {info['size']} ({info['n_examples']} examples)")

    # Stage 4: Reranking
    print("\n--- Stage 4: Cross-Encoder Reranking ---")
    if args.skip_rerank:
        rerank_stats = {
            "total_sec": None,
            "per_example_sec": None,
            "n_examples": 1704,
            "avg_candidates": 208.5,
            "source": "skipped",
        }
        print("  Skipped")
    else:
        rerank_stats = estimate_reranking()
        if rerank_stats["total_sec"]:
            print(f"  Total: ~{format_time(rerank_stats['total_sec'])} "
                  f"({rerank_stats['per_example_sec']:.2f}s/example)")
            print(f"  Source: {rerank_stats['source']}")

    # Print summary table
    print_results(
        cc_stats, imp_stats, n_train, n_rels,
        expansion_stats, pool_sizes, rerank_stats
    )

    # Save JSON
    save_json_stats(
        cc_stats, imp_stats, n_train, n_rels,
        expansion_stats, pool_sizes, rerank_stats, args.output
    )


if __name__ == "__main__":
    main()
