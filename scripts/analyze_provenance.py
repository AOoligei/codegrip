#!/usr/bin/env python3
"""
Candidate provenance analysis: Does graph-hard training specifically help
when the correct file comes from graph expansion (vs BM25 retrieval)?

Stratifies test examples by whether the ground truth file is:
  (a) in the BM25-top-500 candidates (bm25_retrievable)
  (b) only in graph-expanded candidates (expansion_only)
  (c) not in either pool (not_in_pool)
"""
import json
import numpy as np
from pathlib import Path

BASE = Path("experiments")


def load_bm25_candidates():
    """Load BM25-top-500 per example."""
    bm25 = {}
    with open("data/rankft/grepo_test_bm25_top500.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            key = (ex["repo"], ex["issue_id"])
            bm25[key] = set(ex["bm25_candidates"])
    return bm25


def load_predictions(exp_name, eval_subdir):
    p = BASE / exp_name / eval_subdir / "predictions.jsonl"
    if not p.exists():
        return None
    preds = {}
    with open(p) as f:
        for line in f:
            ex = json.loads(line)
            key = (ex["repo"], ex["issue_id"])
            preds[key] = ex
    return preds


def main():
    bm25_cands = load_bm25_candidates()

    seeds = [42, 1, 2, 3, 4]
    graph_exps = {42: "rankft_runB_graph", 1: "rankft_runB_graph_seed1",
                  2: "rankft_runB_graph_seed2", 3: "rankft_runB_graph_seed3",
                  4: "rankft_runB_graph_seed4"}
    bm25_exps = {42: "rankft_runA_bm25only", 1: "rankft_runA_bm25only_seed1",
                 2: "rankft_runA_bm25only_seed2", 3: "rankft_runA_bm25only_seed3",
                 4: "rankft_runA_bm25only_seed4"}

    all_bm25_deltas = []
    all_exp_deltas = []

    for seed in seeds:
        g_preds = load_predictions(graph_exps[seed], "eval_merged_rerank")
        b_preds = load_predictions(bm25_exps[seed], "eval_merged_rerank")
        if g_preds is None or b_preds is None:
            print(f"Seed {seed}: MISSING predictions")
            continue

        strata = {
            "bm25_retrievable": {"g_r1": [], "b_r1": []},
            "expansion_only": {"g_r1": [], "b_r1": []},
            "not_in_pool": {"g_r1": [], "b_r1": []},
        }

        for key in g_preds:
            if key not in b_preds:
                continue
            g = g_preds[key]
            b = b_preds[key]
            gt_set = set(g["ground_truth"])
            bm25_set = bm25_cands.get(key, set())

            # Classify: is any GT in BM25 candidates?
            gt_in_bm25 = bool(gt_set & bm25_set)
            gt_in_pool = g.get("gt_in_candidates", False)

            if not gt_in_pool:
                cat = "not_in_pool"
            elif gt_in_bm25:
                cat = "bm25_retrievable"
            else:
                cat = "expansion_only"

            # Use per-example recall@1 from the prediction metrics
            g_r1 = g["metrics"]["recall@1"]
            b_r1 = b["metrics"]["recall@1"]
            strata[cat]["g_r1"].append(g_r1)
            strata[cat]["b_r1"].append(b_r1)

        print(f"\n=== Seed {seed} ===")
        print(f"{'Stratum':<22} {'N':>5} {'Graph R@1':>10} {'BM25 R@1':>10} {'Delta':>8}")
        print("-" * 60)
        for cat in ["bm25_retrievable", "expansion_only", "not_in_pool"]:
            s = strata[cat]
            n = len(s["g_r1"])
            if n > 0:
                g_mean = np.mean(s["g_r1"]) * 100
                b_mean = np.mean(s["b_r1"]) * 100
                delta = g_mean - b_mean
                print(f"{cat:<22} {n:>5} {g_mean:>9.2f}% {b_mean:>9.2f}% {delta:>+7.2f}%")
                if cat == "bm25_retrievable":
                    all_bm25_deltas.append(delta)
                elif cat == "expansion_only":
                    all_exp_deltas.append(delta)
            else:
                print(f"{cat:<22} {n:>5}       N/A       N/A      N/A")

    if len(all_bm25_deltas) >= 2:
        from scipy import stats
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({len(all_bm25_deltas)} seeds)")
        print(f"{'='*60}")

        bm_mean = np.mean(all_bm25_deltas)
        bm_std = np.std(all_bm25_deltas, ddof=1)
        exp_mean = np.mean(all_exp_deltas)
        exp_std = np.std(all_exp_deltas, ddof=1)

        print(f"BM25-retrievable: graph delta = {bm_mean:+.2f}% +/- {bm_std:.2f}%")
        t1, p1 = stats.ttest_1samp(all_bm25_deltas, 0)
        print(f"  t={t1:.3f}, p={p1:.4f}")

        print(f"Expansion-only:   graph delta = {exp_mean:+.2f}% +/- {exp_std:.2f}%")
        t2, p2 = stats.ttest_1samp(all_exp_deltas, 0)
        print(f"  t={t2:.3f}, p={p2:.4f}")

        # Interaction: does graph help more on expansion-only?
        interactions = [e - b for e, b in zip(all_exp_deltas, all_bm25_deltas)]
        int_mean = np.mean(interactions)
        int_std = np.std(interactions, ddof=1)
        print(f"\nProvenance interaction (expansion - bm25 delta): {int_mean:+.2f}% +/- {int_std:.2f}%")
        if len(interactions) >= 3:
            t3, p3 = stats.ttest_1samp(interactions, 0)
            print(f"  t={t3:.3f}, p={p3:.4f}")


if __name__ == "__main__":
    main()
