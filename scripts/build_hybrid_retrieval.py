#!/usr/bin/env python3
"""
Build hybrid BM25 + SweRank dense retrieval baseline for CodeGRIP.

Pipeline:
  Step 1: Run SweRank full retrieval from all files, save top-500 per example.
  Step 2: Fuse BM25 top-500 and SweRank top-500 via Reciprocal Rank Fusion (RRF).
  Step 3: Apply graph expansion to hybrid top-500 (same method as merged_bm25_exp6).
  Step 4: Print oracle recall comparison.

Usage:
    # Full pipeline (step 1 is expensive, ~30 min on 1 GPU)
    python scripts/build_hybrid_retrieval.py --device cuda:6

    # If SweRank top-500 already exists, skip step 1
    python scripts/build_hybrid_retrieval.py --skip_dense

    # Dry run: just print what would happen
    python scripts/build_hybrid_retrieval.py --dry_run
"""
import json
import os
import sys
import argparse
import random
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional

import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BM25_TOP500 = os.path.join(PROJECT_ROOT, "data/rankft/grepo_test_bm25_top500.jsonl")
TEST_DATA = os.path.join(PROJECT_ROOT, "data/grepo_text/grepo_test.jsonl")
TRAIN_DATA = os.path.join(PROJECT_ROOT, "data/grepo_text/grepo_train.jsonl")
FILE_TREE_DIR = os.path.join(PROJECT_ROOT, "data/file_trees")
REPOS_DIR = os.path.join(PROJECT_ROOT, "data/repos")
DEP_GRAPH_DIR = os.path.join(PROJECT_ROOT, "data/dep_graphs")

# Outputs
DENSE_TOP500 = os.path.join(PROJECT_ROOT, "data/rankft/grepo_test_swerank_top500.jsonl")
HYBRID_TOP500 = os.path.join(PROJECT_ROOT, "data/rankft/grepo_test_hybrid_top500.jsonl")
MERGED_HYBRID_GRAPH = os.path.join(PROJECT_ROOT, "data/rankft/merged_hybrid_graph_candidates.jsonl")


# ============================================================
# Step 1: Dense retrieval (top-500 from all files)
# ============================================================
DENSE_MODELS = {
    "e5-large": {
        "hf_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "max_seq_length": 512,
    },
    "swerank": {
        "hf_name": "/home/chenlibin/models/SweRankEmbed-Large",
        "query_prefix": (
            "Instruct: Given a github issue, identify the code that needs to "
            "be changed to fix the issue.\nQuery: "
        ),
        "doc_prefix": "",
        "max_seq_length": 1024,
    },
}


def run_dense_retrieval(device: str, batch_size: int, max_lines: int,
                        model_key: str = "e5-large"):
    """Run dense model on all py_files per repo, save top-500."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)

    config = DENSE_MODELS[model_key]

    print("=" * 60)
    print(f"Step 1: {model_key} Dense Retrieval (top-500 from all files)")
    print("=" * 60)

    # Load model
    print(f"Loading {config['hf_name']} on {device}...")
    model = SentenceTransformer(
        config["hf_name"],
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = config["max_seq_length"]
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded ({n_params:.0f}M params)")

    # Load test data
    print("Loading test data and file trees...")
    examples = []
    with open(TEST_DATA) as f:
        for line in f:
            examples.append(json.loads(line))

    file_trees = {}
    for fname in os.listdir(FILE_TREE_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(FILE_TREE_DIR, fname)) as f:
                tree = json.load(f)
            file_trees[tree["repo"]] = tree

    # Group by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo = ex["repo"]
        gt_files = set(ex.get("changed_py_files", []))
        if gt_files and repo in file_trees:
            repo_examples[repo].append(ex)

    total = sum(len(v) for v in repo_examples.values())
    print(f"  {total} examples across {len(repo_examples)} repos")

    # Process per repo
    results = []
    processed = 0
    t0 = time.time()

    for repo_idx, (repo, exs) in enumerate(sorted(repo_examples.items())):
        tree = file_trees[repo]
        py_files = tree["py_files"]

        # Read file contents and encode
        repo_dir = os.path.join(REPOS_DIR, repo)
        doc_texts = []
        valid_files = []

        for fp in py_files:
            full_path = os.path.join(repo_dir, fp)
            try:
                with open(full_path, "r", errors="replace") as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line.rstrip())
                content = "\n".join(lines)
            except (FileNotFoundError, PermissionError):
                content = ""

            doc_text = f"{fp}\n{content}" if content else fp
            doc_texts.append(f"{config['doc_prefix']}{doc_text}")
            valid_files.append(fp)

        if not doc_texts:
            continue

        repo_t0 = time.time()
        with torch.no_grad():
            file_embs = model.encode(
                doc_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        for ex in exs:
            query_text = f"{config['query_prefix']}{ex['issue_text']}"
            with torch.no_grad():
                query_emb = model.encode(
                    [query_text],
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                scores = torch.matmul(query_emb, file_embs.T).squeeze(0)
                top_k = min(500, len(valid_files))
                top_indices = torch.argsort(scores, descending=True)[:top_k].cpu().numpy()

            ranked_files = [valid_files[i] for i in top_indices]
            gt_files = set(ex.get("changed_py_files", []))

            results.append({
                "repo": ex["repo"],
                "issue_id": ex["issue_id"],
                "issue_text": ex["issue_text"],
                "ground_truth": list(gt_files),
                "dense_candidates": ranked_files,
            })
            processed += 1

        repo_time = time.time() - repo_t0
        if (repo_idx + 1) % 5 == 0 or repo_idx == len(repo_examples) - 1:
            elapsed = time.time() - t0
            print(f"  [{processed}/{total}] {repo} ({len(py_files)} files, "
                  f"{repo_time:.1f}s) | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} examples in {elapsed:.0f}s")

    # Save
    with open(DENSE_TOP500, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {DENSE_TOP500}")

    # Clean up GPU memory
    del model, file_embs
    torch.cuda.empty_cache()

    return results


# ============================================================
# Step 2: RRF Fusion
# ============================================================
def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
    top_n: int = 500,
) -> List[str]:
    """
    Reciprocal Rank Fusion of multiple ranked lists.
    RRF score = sum(1 / (k + rank)) across all lists.
    rank is 1-based.
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank_0based, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank_0based + 1)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items[:top_n]]


def run_rrf_fusion():
    """Fuse BM25 top-500 and SweRank top-500 via RRF."""
    print("\n" + "=" * 60)
    print("Step 2: Reciprocal Rank Fusion (BM25 + SweRank)")
    print("=" * 60)

    # Load BM25 candidates
    bm25_data = {}
    with open(BM25_TOP500) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            bm25_data[key] = d

    # Load dense candidates
    dense_data = {}
    with open(DENSE_TOP500) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            dense_data[key] = d

    print(f"  BM25 examples: {len(bm25_data)}")
    print(f"  Dense examples: {len(dense_data)}")

    # Match keys
    common_keys = set(bm25_data.keys()) & set(dense_data.keys())
    bm25_only = set(bm25_data.keys()) - common_keys
    dense_only = set(dense_data.keys()) - common_keys
    print(f"  Common: {len(common_keys)}, BM25-only: {len(bm25_only)}, Dense-only: {len(dense_only)}")

    results = []
    for key in sorted(common_keys):
        bm25_item = bm25_data[key]
        dense_item = dense_data[key]

        bm25_ranking = bm25_item["bm25_candidates"]
        dense_ranking = dense_item["dense_candidates"]

        hybrid_ranking = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking], k=60, top_n=500
        )

        results.append({
            "repo": bm25_item["repo"],
            "issue_id": bm25_item["issue_id"],
            "issue_text": bm25_item["issue_text"],
            "ground_truth": bm25_item["ground_truth"],
            "bm25_candidates": bm25_ranking,
            "hybrid_candidates": hybrid_ranking,
        })

    # For BM25-only examples (dense retrieval may have missed them), keep BM25 only
    for key in sorted(bm25_only):
        bm25_item = bm25_data[key]
        results.append({
            "repo": bm25_item["repo"],
            "issue_id": bm25_item["issue_id"],
            "issue_text": bm25_item["issue_text"],
            "ground_truth": bm25_item["ground_truth"],
            "bm25_candidates": bm25_item["bm25_candidates"],
            "hybrid_candidates": bm25_item["bm25_candidates"],  # fallback to BM25
        })

    # Save
    with open(HYBRID_TOP500, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {HYBRID_TOP500} ({len(results)} examples)")

    return results


# ============================================================
# Step 3: Graph expansion on hybrid candidates
# ============================================================
def build_cochange_index(min_cochange: int = 1) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build co-change index from training data."""
    repo_cochanges: Dict[str, Counter] = defaultdict(Counter)
    repo_file_count: Dict[str, Counter] = defaultdict(Counter)

    with open(TRAIN_DATA) as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") != "train":
                continue
            repo = item["repo"]
            files = item.get("changed_py_files", [])
            if not files:
                files = [fp for fp in item.get("changed_files", []) if fp.endswith(".py")]
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


def build_import_index() -> Dict[str, Dict[str, Set[str]]]:
    """Build import graph index from dep_graphs."""
    index: Dict[str, Dict[str, Set[str]]] = {}
    if not os.path.isdir(DEP_GRAPH_DIR):
        return index

    for fname in os.listdir(DEP_GRAPH_DIR):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(DEP_GRAPH_DIR, fname)) as f:
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


def expand_with_cochange(
    seed_files: List[str],
    cochange_index: Dict[str, Dict[str, float]],
    max_expand: int = 10,
    min_score: float = 0.05,
) -> List[str]:
    """Get co-change neighbors for seed files."""
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
    """Get import neighbors for seed files."""
    scores: Dict[str, int] = defaultdict(int)
    for pred_file in seed_files:
        for neighbor in import_index.get(pred_file, set()):
            if neighbor not in exclude:
                scores[neighbor] += 1
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_cands[:max_expand]]


def run_graph_expansion():
    """Apply graph expansion to hybrid top-500 candidates."""
    print("\n" + "=" * 60)
    print("Step 3: Graph Expansion on Hybrid Candidates")
    print("=" * 60)

    # Build indices
    print("  Building co-change index...")
    cc_index = build_cochange_index()
    cc_repos = len(cc_index)
    print(f"    {cc_repos} repos with co-change data")

    print("  Building import index...")
    imp_index = build_import_index()
    imp_repos = len(imp_index)
    print(f"    {imp_repos} repos with import data")

    # Load hybrid candidates
    hybrid_data = []
    with open(HYBRID_TOP500) as f:
        for line in f:
            hybrid_data.append(json.loads(line))
    print(f"  Loaded {len(hybrid_data)} hybrid examples")

    # Expand
    results = []
    n_expanded = 0
    total_graph_additions = 0

    for item in hybrid_data:
        repo = item["repo"]
        hybrid_cands = item["hybrid_candidates"]

        # Use top-20 as seed (same as create_edge_type_pools.py)
        seed = hybrid_cands[:20]

        # Co-change expansion
        repo_cc = cc_index.get(repo, {})
        cc_expansion = expand_with_cochange(seed, repo_cc, max_expand=10, min_score=0.05)

        # Import expansion
        already = set(seed) | set(cc_expansion)
        repo_imp = imp_index.get(repo, {})
        imp_expansion = expand_with_imports(seed, repo_imp, exclude=already, max_expand=10)

        # Merge: graph neighbors first, then hybrid candidates fill (deduped)
        graph_neighbors = cc_expansion + imp_expansion
        seen = set()
        merged = []
        for c in graph_neighbors:
            if c not in seen:
                merged.append(c)
                seen.add(c)
        for c in hybrid_cands:
            if c not in seen:
                merged.append(c)
                seen.add(c)
        merged = merged[:len(hybrid_cands)]  # truncate to same size

        n_graph = sum(1 for c in merged if c not in set(hybrid_cands))
        if n_graph > 0:
            n_expanded += 1
        total_graph_additions += n_graph

        results.append({
            "repo": repo,
            "issue_id": item["issue_id"],
            "candidates": merged,
        })

    avg_graph = total_graph_additions / len(results) if results else 0
    print(f"  Expanded: {n_expanded}/{len(results)} examples "
          f"({n_expanded/len(results)*100:.1f}%)")
    print(f"  Avg graph additions: {avg_graph:.1f}")

    # Save
    with open(MERGED_HYBRID_GRAPH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved: {MERGED_HYBRID_GRAPH}")

    return results


# ============================================================
# Step 4: Oracle recall comparison
# ============================================================
def compute_oracle_recall(candidates: List[str], ground_truth: Set[str]) -> float:
    """Oracle recall: fraction of GT files appearing anywhere in candidates."""
    if not ground_truth:
        return 0.0
    return len(set(candidates) & ground_truth) / len(ground_truth) * 100


def run_comparison():
    """Print oracle recall comparison across all settings."""
    print("\n" + "=" * 60)
    print("Step 4: Oracle Recall Comparison")
    print("=" * 60)

    # Load test GT
    gt_data = {}
    with open(TEST_DATA) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            gt_data[key] = set(d.get("changed_py_files", []))

    settings = {}

    # BM25 top-500
    if os.path.exists(BM25_TOP500):
        recalls = []
        with open(BM25_TOP500) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                gt = gt_data.get(key, set())
                if gt:
                    recalls.append(compute_oracle_recall(d["bm25_candidates"], gt))
        settings["BM25-500"] = recalls

    # Dense top-500
    if os.path.exists(DENSE_TOP500):
        recalls = []
        with open(DENSE_TOP500) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                gt = gt_data.get(key, set())
                if gt:
                    recalls.append(compute_oracle_recall(d["dense_candidates"], gt))
        settings["SweRank-500"] = recalls

    # Hybrid top-500
    if os.path.exists(HYBRID_TOP500):
        recalls = []
        with open(HYBRID_TOP500) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                gt = gt_data.get(key, set())
                if gt:
                    recalls.append(compute_oracle_recall(d["hybrid_candidates"], gt))
        settings["Hybrid-500 (RRF)"] = recalls

    # BM25 + Graph (existing merged_bm25_exp6)
    merged_bm25_graph = os.path.join(PROJECT_ROOT, "data/rankft/merged_bm25_exp6_candidates.jsonl")
    if os.path.exists(merged_bm25_graph):
        recalls = []
        with open(merged_bm25_graph) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                gt = gt_data.get(key, set())
                if gt and key in gt_data:
                    # Only count test examples
                    if key in {(dd["repo"], dd["issue_id"]) for dd in []}:
                        pass
                    recalls.append(compute_oracle_recall(d["candidates"], gt))
        # Re-do: only test keys
        recalls = []
        with open(merged_bm25_graph) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                if key not in gt_data:
                    continue
                gt = gt_data[key]
                if gt:
                    recalls.append(compute_oracle_recall(d["candidates"], gt))
        settings["BM25 + Graph"] = recalls

    # Hybrid + Graph
    if os.path.exists(MERGED_HYBRID_GRAPH):
        recalls = []
        with open(MERGED_HYBRID_GRAPH) as f:
            for line in f:
                d = json.loads(line)
                key = (d["repo"], d["issue_id"])
                gt = gt_data.get(key, set())
                if gt:
                    recalls.append(compute_oracle_recall(d["candidates"], gt))
        settings["Hybrid + Graph"] = recalls

    # Print comparison table
    print(f"\n{'Setting':<25} {'N':>6} {'Oracle Recall':>14} {'Median':>8}")
    print("-" * 58)
    for name, recalls in settings.items():
        if recalls:
            mean_r = np.mean(recalls)
            median_r = np.median(recalls)
            print(f"{name:<25} {len(recalls):>6} {mean_r:>13.2f}% {median_r:>7.2f}%")

    # Pairwise deltas
    if "BM25-500" in settings and "Hybrid-500 (RRF)" in settings:
        bm25_r = settings["BM25-500"]
        hybrid_r = settings["Hybrid-500 (RRF)"]
        if len(bm25_r) == len(hybrid_r):
            deltas = [h - b for h, b in zip(hybrid_r, bm25_r)]
            print(f"\n  Hybrid vs BM25: mean delta = {np.mean(deltas):+.2f}%")
            print(f"  Improved: {sum(1 for d in deltas if d > 0)}/{len(deltas)}, "
                  f"Hurt: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")

    if "BM25 + Graph" in settings and "Hybrid + Graph" in settings:
        bm25g_r = settings["BM25 + Graph"]
        hybridg_r = settings["Hybrid + Graph"]
        if len(bm25g_r) == len(hybridg_r):
            deltas = [h - b for h, b in zip(hybridg_r, bm25g_r)]
            print(f"\n  Hybrid+Graph vs BM25+Graph: mean delta = {np.mean(deltas):+.2f}%")

    if "Hybrid-500 (RRF)" in settings and "Hybrid + Graph" in settings:
        hybrid_r = settings["Hybrid-500 (RRF)"]
        hybridg_r = settings["Hybrid + Graph"]
        if len(hybrid_r) == len(hybridg_r):
            deltas = [h - b for h, b in zip(hybridg_r, hybrid_r)]
            print(f"\n  Graph expansion on Hybrid: mean delta = {np.mean(deltas):+.2f}%")
            print(f"  Improved: {sum(1 for d in deltas if d > 0)}/{len(deltas)}, "
                  f"Hurt: {sum(1 for d in deltas if d < 0)}/{len(deltas)}")

    if "BM25-500" in settings and "BM25 + Graph" in settings:
        bm25_r = settings["BM25-500"]
        bm25g_r = settings["BM25 + Graph"]
        if len(bm25_r) == len(bm25g_r):
            deltas = [h - b for h, b in zip(bm25g_r, bm25_r)]
            print(f"\n  Graph expansion on BM25: mean delta = {np.mean(deltas):+.2f}%")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Build hybrid BM25 + dense retrieval baseline")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--dense_model", default="e5-large",
                        choices=list(DENSE_MODELS.keys()),
                        help="Dense retrieval model to use")
    parser.add_argument("--skip_dense", action="store_true",
                        help="Skip step 1 (dense retrieval), use existing file")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan only, do not execute")
    args = parser.parse_args()

    # Override output paths based on dense model choice
    global DENSE_TOP500, HYBRID_TOP500, MERGED_HYBRID_GRAPH
    model_tag = args.dense_model.replace("-", "")
    DENSE_TOP500 = os.path.join(PROJECT_ROOT, f"data/rankft/grepo_test_{model_tag}_top500.jsonl")
    HYBRID_TOP500 = os.path.join(PROJECT_ROOT, f"data/rankft/grepo_test_hybrid_{model_tag}_top500.jsonl")
    MERGED_HYBRID_GRAPH = os.path.join(PROJECT_ROOT, f"data/rankft/merged_hybrid_{model_tag}_graph_candidates.jsonl")

    if args.dry_run:
        print("DRY RUN: Would execute:")
        print(f"  Step 1: {args.dense_model} full retrieval -> {DENSE_TOP500}")
        print(f"  Step 2: RRF fusion -> {HYBRID_TOP500}")
        print(f"  Step 3: Graph expansion -> {MERGED_HYBRID_GRAPH}")
        print(f"  Step 4: Oracle recall comparison")
        return

    t_total = time.time()

    # Step 1
    if args.skip_dense:
        if not os.path.exists(DENSE_TOP500):
            print(f"ERROR: --skip_dense but {DENSE_TOP500} does not exist")
            sys.exit(1)
        print(f"Skipping step 1, using existing: {DENSE_TOP500}")
    else:
        run_dense_retrieval(args.device, args.batch_size, args.max_lines,
                            model_key=args.dense_model)

    # Step 2
    run_rrf_fusion()

    # Step 3
    run_graph_expansion()

    # Step 4
    run_comparison()

    total_elapsed = time.time() - t_total
    print(f"\nTotal pipeline time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
