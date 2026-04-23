#!/usr/bin/env python3
"""
Score calibration analysis: Compare how graph-hard vs BM25-only models
score GT files and top negatives. Does graph-hard training change the
score distribution in ways that explain the conditional benefit?
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path("experiments")


def load_predictions(exp_name, eval_subdir):
    p = BASE / exp_name / eval_subdir / "predictions.jsonl"
    if not p.exists():
        return None
    preds = []
    with open(p) as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def analyze_scores(preds):
    """Extract score statistics for GT and top negatives."""
    gt_scores = []
    top_neg_scores = []
    score_gaps = []  # GT score - top negative score
    gt_ranks = []

    for ex in preds:
        gt_set = set(ex["ground_truth"])
        predicted = ex["predicted"]
        scores = ex["scores"]

        # Find GT scores and ranks
        for i, (f, s) in enumerate(zip(predicted, scores)):
            if f in gt_set:
                gt_scores.append(s)
                gt_ranks.append(i + 1)
                break  # first GT occurrence

        # Top negative: highest-scored non-GT file
        for i, (f, s) in enumerate(zip(predicted, scores)):
            if f not in gt_set:
                top_neg_scores.append(s)
                break

        # Score gap: GT_score - top_neg_score
        gt_s = None
        neg_s = None
        for f, s in zip(predicted, scores):
            if f in gt_set and gt_s is None:
                gt_s = s
            if f not in gt_set and neg_s is None:
                neg_s = s
            if gt_s is not None and neg_s is not None:
                break
        if gt_s is not None and neg_s is not None:
            score_gaps.append(gt_s - neg_s)

    return {
        "gt_score_mean": np.mean(gt_scores) if gt_scores else 0,
        "gt_score_std": np.std(gt_scores) if gt_scores else 0,
        "neg_score_mean": np.mean(top_neg_scores) if top_neg_scores else 0,
        "neg_score_std": np.std(top_neg_scores) if top_neg_scores else 0,
        "gap_mean": np.mean(score_gaps) if score_gaps else 0,
        "gap_std": np.std(score_gaps) if score_gaps else 0,
        "median_gt_rank": np.median(gt_ranks) if gt_ranks else 0,
        "n_gt_found": len(gt_scores),
    }


def main():
    seeds = [42, 1, 2, 4]
    graph_exps = {42: "rankft_runB_graph", 1: "rankft_runB_graph_seed1",
                  2: "rankft_runB_graph_seed2", 4: "rankft_runB_graph_seed4"}
    bm25_exps = {42: "rankft_runA_bm25only", 1: "rankft_runA_bm25only_seed1",
                 2: "rankft_runA_bm25only_seed2", 4: "rankft_runA_bm25only_seed4"}

    print("=== Score Calibration: Expanded Pool ===\n")
    print(f"{'Seed':<6} {'Model':<8} {'GT score':>10} {'Neg score':>10} {'Gap':>10} {'Med rank':>10}")
    print("-" * 62)

    for seed in seeds:
        for label, exps in [("Graph", graph_exps), ("BM25", bm25_exps)]:
            preds = load_predictions(exps[seed], "eval_merged_rerank")
            if preds is None:
                continue
            stats = analyze_scores(preds)
            print(f"{seed:<6} {label:<8} {stats['gt_score_mean']:>9.2f} {stats['neg_score_mean']:>9.2f} "
                  f"{stats['gap_mean']:>+9.2f} {stats['median_gt_rank']:>9.0f}")

    # BM25 pool analysis (where available)
    print("\n=== Score Calibration: BM25 Pool ===\n")
    print(f"{'Seed':<6} {'Model':<8} {'GT score':>10} {'Neg score':>10} {'Gap':>10} {'Med rank':>10}")
    print("-" * 62)

    for seed in seeds:
        for label, exps in [("Graph", graph_exps), ("BM25", bm25_exps)]:
            preds = load_predictions(exps[seed], "eval_bm25pool")
            if preds is None:
                continue
            stats = analyze_scores(preds)
            print(f"{seed:<6} {label:<8} {stats['gt_score_mean']:>9.2f} {stats['neg_score_mean']:>9.2f} "
                  f"{stats['gap_mean']:>+9.2f} {stats['median_gt_rank']:>9.0f}")

    # Aggregate: does graph model suppress negatives more?
    print("\n=== Aggregate Score Gap (GT - top negative) ===\n")
    for pool, subdir in [("Expanded", "eval_merged_rerank"), ("BM25", "eval_bm25pool")]:
        g_gaps = []
        b_gaps = []
        for seed in seeds:
            g_preds = load_predictions(graph_exps[seed], subdir)
            b_preds = load_predictions(bm25_exps[seed], subdir)
            if g_preds:
                gs = analyze_scores(g_preds)
                g_gaps.append(gs["gap_mean"])
            if b_preds:
                bs = analyze_scores(b_preds)
                b_gaps.append(bs["gap_mean"])
        if g_gaps:
            print(f"{pool} pool - Graph gap: {np.mean(g_gaps):+.2f} ± {np.std(g_gaps, ddof=1):.2f} ({len(g_gaps)} seeds)")
        if b_gaps:
            print(f"{pool} pool - BM25  gap: {np.mean(b_gaps):+.2f} ± {np.std(b_gaps, ddof=1):.2f} ({len(b_gaps)} seeds)")


if __name__ == "__main__":
    main()
