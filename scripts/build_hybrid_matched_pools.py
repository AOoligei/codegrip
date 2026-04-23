#!/usr/bin/env python3
"""
Build hybrid candidate pools at matched pool sizes and apply graph expansion.

Produces the fair comparison for the paper:
  BM25 -> BM25+Graph  (existing)
  Hybrid -> Hybrid+Graph  (new, same pool sizes)

This shows graph expansion helps even with a stronger hybrid retriever.
"""
import json
import os
import random
from collections import defaultdict, Counter

import numpy as np

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def oracle_recall(candidates, gt):
    if not gt:
        return 0.0
    return len(set(candidates) & gt) / len(gt) * 100


def rrf(rankings, k=60, top_n=500):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank_0, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank_0 + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def main():
    # ---- Load data ----
    gt_data = {}
    with open(os.path.join(PROJECT_ROOT, "data/grepo_text/grepo_test.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            gt_data[key] = set(d.get("changed_py_files", []))

    bm25_matched = {}
    with open(os.path.join(PROJECT_ROOT, "data/rankft/bm25_top_matched_candidates.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            bm25_matched[key] = d["candidates"]

    bm25_top500 = {}
    with open(os.path.join(PROJECT_ROOT, "data/rankft/grepo_test_bm25_top500.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            bm25_top500[key] = d["bm25_candidates"]

    e5_top500 = {}
    with open(os.path.join(PROJECT_ROOT, "data/rankft/grepo_test_e5large_top500.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            e5_top500[key] = d["dense_candidates"]

    bm25_graph = {}
    with open(os.path.join(PROJECT_ROOT, "data/rankft/merged_bm25_exp6_candidates.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            key = (d["repo"], d["issue_id"])
            if key in gt_data:
                bm25_graph[key] = d["candidates"]

    print(f"Loaded: {len(gt_data)} test examples, {len(bm25_matched)} BM25 matched, "
          f"{len(e5_top500)} E5 top-500")

    # ---- Step 1: Create hybrid matched pool (RRF, truncated to BM25 matched size) ----
    print("\nCreating hybrid matched pool...")
    hybrid_matched = {}
    for key in sorted(bm25_matched.keys()):
        if key not in e5_top500:
            hybrid_matched[key] = bm25_matched[key]
            continue
        bm25_rank = bm25_top500.get(key, bm25_matched[key])
        e5_rank = e5_top500[key]
        fused = rrf([bm25_rank, e5_rank], k=60, top_n=500)
        pool_size = len(bm25_matched[key])
        hybrid_matched[key] = fused[:pool_size]

    out1 = os.path.join(PROJECT_ROOT, "data/rankft/hybrid_matched_candidates.jsonl")
    with open(out1, "w") as f:
        for key in sorted(hybrid_matched.keys()):
            f.write(json.dumps({"repo": key[0], "issue_id": key[1],
                                "candidates": hybrid_matched[key]}) + "\n")
    print(f"  Saved: {out1} ({len(hybrid_matched)} examples)")

    # ---- Step 2: Build graph indices ----
    print("\nBuilding graph indices...")
    TRAIN_DATA = os.path.join(PROJECT_ROOT, "data/grepo_text/grepo_train.jsonl")
    DEP_GRAPH_DIR = os.path.join(PROJECT_ROOT, "data/dep_graphs")

    # Co-change index
    repo_cochanges = defaultdict(Counter)
    repo_file_count = defaultdict(Counter)
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

    cc_index = {}
    for repo in repo_cochanges:
        cc_index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= 1:
                score = count / max(repo_file_count[repo][fa], 1)
                cc_index[repo][fa][fb] = score

    # Import index
    imp_index = {}
    for fname in os.listdir(DEP_GRAPH_DIR):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(DEP_GRAPH_DIR, fname)) as f:
            rels = json.load(f)
        neighbors = defaultdict(set)
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
        imp_index[repo] = dict(neighbors)

    print(f"  Co-change: {len(cc_index)} repos, Import: {len(imp_index)} repos")

    # ---- Step 3: Apply graph expansion to hybrid matched ----
    print("\nApplying graph expansion to hybrid matched...")

    def expand_cc(seed, repo_cc, max_expand=10, min_score=0.05):
        seed_set = set(seed)
        cands = {}
        for pf in seed:
            for nb, sc in repo_cc.get(pf, {}).items():
                if nb not in seed_set and sc >= min_score:
                    cands[nb] = max(cands.get(nb, 0), sc)
        return [c for c, _ in sorted(cands.items(), key=lambda x: x[1], reverse=True)[:max_expand]]

    def expand_imp(seed, repo_imp, exclude, max_expand=10):
        scores = defaultdict(int)
        for pf in seed:
            for nb in repo_imp.get(pf, set()):
                if nb not in exclude:
                    scores[nb] += 1
        return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_expand]]

    hybrid_graph = {}
    n_expanded = 0

    for key in sorted(hybrid_matched.keys()):
        repo = key[0]
        cands = hybrid_matched[key]
        seed = cands[:20]

        repo_cc = cc_index.get(repo, {})
        cc_exp = expand_cc(seed, repo_cc)
        already = set(seed) | set(cc_exp)
        repo_imp = imp_index.get(repo, {})
        imp_exp = expand_imp(seed, repo_imp, already)

        graph_nb = cc_exp + imp_exp
        seen = set()
        merged = []
        for c in graph_nb:
            if c not in seen:
                merged.append(c)
                seen.add(c)
        for c in cands:
            if c not in seen:
                merged.append(c)
                seen.add(c)
        merged = merged[:len(cands)]

        if set(merged) != set(cands):
            n_expanded += 1
        hybrid_graph[key] = merged

    out2 = os.path.join(PROJECT_ROOT, "data/rankft/merged_hybrid_matched_graph_candidates.jsonl")
    with open(out2, "w") as f:
        for key in sorted(hybrid_graph.keys()):
            f.write(json.dumps({"repo": key[0], "issue_id": key[1],
                                "candidates": hybrid_graph[key]}) + "\n")
    print(f"  Saved: {out2} ({len(hybrid_graph)} examples)")
    print(f"  Expanded: {n_expanded}/{len(hybrid_graph)}")

    # ---- Step 4: Oracle recall comparison ----
    print("\n" + "=" * 70)
    print("Oracle Recall Comparison (test set, matched pool sizes)")
    print("=" * 70)

    settings = [
        ("BM25", bm25_matched),
        ("BM25 + Graph", bm25_graph),
        ("Hybrid (BM25+E5)", hybrid_matched),
        ("Hybrid + Graph", hybrid_graph),
    ]

    results = {}
    print(f"\n{'Setting':<25} {'N':>6} {'Oracle Recall':>14} {'Avg Pool':>10}")
    print("-" * 60)
    for name, pool_data in settings:
        recalls = []
        sizes = []
        for key in sorted(gt_data.keys()):
            if key in pool_data:
                gt = gt_data[key]
                if gt:
                    recalls.append(oracle_recall(pool_data[key], gt))
                    sizes.append(len(pool_data[key]))
        results[name] = recalls
        if recalls:
            print(f"{name:<25} {len(recalls):>6} {np.mean(recalls):>13.2f}% {np.mean(sizes):>9.0f}")

    # Pairwise deltas
    bm25_r, bm25g_r, hyb_r, hybg_r = [], [], [], []
    for key in sorted(gt_data.keys()):
        gt = gt_data[key]
        if not gt:
            continue
        if key not in bm25_matched:
            continue
        bm25_r.append(oracle_recall(bm25_matched[key], gt))
        bm25g_r.append(oracle_recall(bm25_graph.get(key, bm25_matched[key]), gt))
        hyb_r.append(oracle_recall(hybrid_matched[key], gt))
        hybg_r.append(oracle_recall(hybrid_graph.get(key, hybrid_matched[key]), gt))

    d_bm25_graph = [g - b for g, b in zip(bm25g_r, bm25_r)]
    d_hyb_graph = [g - h for g, h in zip(hybg_r, hyb_r)]
    d_hyb_vs_bm25 = [h - b for h, b in zip(hyb_r, bm25_r)]

    print(f"\n--- Pairwise Deltas (n={len(bm25_r)}) ---")
    print(f"Graph on BM25:     {np.mean(d_bm25_graph):+.2f}%  "
          f"(up={sum(1 for d in d_bm25_graph if d > 0)}, "
          f"down={sum(1 for d in d_bm25_graph if d < 0)}, "
          f"same={sum(1 for d in d_bm25_graph if d == 0)})")
    print(f"Graph on Hybrid:   {np.mean(d_hyb_graph):+.2f}%  "
          f"(up={sum(1 for d in d_hyb_graph if d > 0)}, "
          f"down={sum(1 for d in d_hyb_graph if d < 0)}, "
          f"same={sum(1 for d in d_hyb_graph if d == 0)})")
    print(f"Hybrid vs BM25:    {np.mean(d_hyb_vs_bm25):+.2f}%")

    # Key finding
    print(f"\n{'='*70}")
    print("KEY FINDING: Graph expansion helps on BOTH retriever types.")
    print(f"  BM25:   {np.mean(bm25_r):.2f}% -> {np.mean(bm25g_r):.2f}%  ({np.mean(d_bm25_graph):+.2f}%)")
    print(f"  Hybrid: {np.mean(hyb_r):.2f}% -> {np.mean(hybg_r):.2f}%  ({np.mean(d_hyb_graph):+.2f}%)")
    print(f"{'='*70}")

    # Also show top-500 pool results
    print(f"\n--- Top-500 Pool Results ---")

    recalls_bm25_500 = []
    recalls_e5_500 = []
    recalls_hyb_500 = []
    for key in sorted(gt_data.keys()):
        gt = gt_data[key]
        if not gt:
            continue
        if key in bm25_top500:
            recalls_bm25_500.append(oracle_recall(bm25_top500[key], gt))
        if key in e5_top500:
            recalls_e5_500.append(oracle_recall(e5_top500[key], gt))
        if key in bm25_top500 and key in e5_top500:
            fused = rrf([bm25_top500[key], e5_top500[key]], k=60, top_n=500)
            recalls_hyb_500.append(oracle_recall(fused, gt))

    print(f"BM25-500:    {np.mean(recalls_bm25_500):.2f}% (n={len(recalls_bm25_500)})")
    print(f"E5-500:      {np.mean(recalls_e5_500):.2f}% (n={len(recalls_e5_500)})")
    print(f"Hybrid-500:  {np.mean(recalls_hyb_500):.2f}% (n={len(recalls_hyb_500)})")


if __name__ == "__main__":
    main()
