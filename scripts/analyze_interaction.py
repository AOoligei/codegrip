#!/usr/bin/env python3
"""
Analyze the 2x2 interaction (pool × reranker) across seeds.
Tests whether graph-hard negatives specifically help on graph-expanded pools.

For each seed, compute:
  delta_expanded = graph_R@1 - bm25_R@1  (on expanded pool)
  delta_bm25pool = graph_R@1 - bm25_R@1  (on BM25-only pool)
  interaction = delta_expanded - delta_bm25pool

If interaction > 0 consistently across seeds, graph-hard negatives
specifically benefit the expanded-pool setting.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path("experiments")


def load_r1(exp_name, eval_subdir):
    p = BASE / exp_name / eval_subdir / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["overall"]["recall@1"]


def main():
    seeds = [42, 1, 2, 3, 4]
    results = []

    for seed in seeds:
        if seed == 42:
            g_exp = "rankft_runB_graph"
            b_exp = "rankft_runA_bm25only"
        else:
            g_exp = f"rankft_runB_graph_seed{seed}"
            b_exp = f"rankft_runA_bm25only_seed{seed}"

        # Expanded pool
        g_expanded = load_r1(g_exp, "eval_merged_rerank")
        b_expanded = load_r1(b_exp, "eval_merged_rerank")

        # BM25-only pool (grepo_test_bm25_top500.jsonl, top_k=200)
        # All seeds use eval_bm25pool for consistency
        # seed42 also has eval_bm25_only_k200 as fallback for graph model
        g_bm25pool = load_r1(g_exp, "eval_bm25pool")
        if g_bm25pool is None and seed == 42:
            g_bm25pool = load_r1(g_exp, "eval_bm25_only_k200")
        b_bm25pool = load_r1(b_exp, "eval_bm25pool")

        has_expanded = g_expanded is not None and b_expanded is not None
        has_bm25pool = g_bm25pool is not None and b_bm25pool is not None

        if has_expanded and has_bm25pool:
            delta_exp = g_expanded - b_expanded
            delta_bm25 = g_bm25pool - b_bm25pool
            interaction = delta_exp - delta_bm25
            results.append({
                "seed": seed,
                "g_exp": g_expanded, "b_exp": b_expanded,
                "g_bm25": g_bm25pool, "b_bm25": b_bm25pool,
                "delta_exp": delta_exp, "delta_bm25": delta_bm25,
                "interaction": interaction,
            })
            print(f"Seed {seed:2d}: expanded Δ={delta_exp:+.2f}  bm25pool Δ={delta_bm25:+.2f}  interaction={interaction:+.2f}")
        else:
            missing = []
            if not has_expanded:
                missing.append("expanded")
            if not has_bm25pool:
                missing.append("bm25pool")
            print(f"Seed {seed:2d}: MISSING {', '.join(missing)}")

    if len(results) < 2:
        print(f"\nOnly {len(results)} seed(s) complete. Need ≥2 for statistics.")
        return

    interactions = [r["interaction"] for r in results]
    mean_int = np.mean(interactions)
    std_int = np.std(interactions, ddof=1)
    t_stat, p_val = stats.ttest_1samp(interactions, 0)

    print(f"\n{'='*60}")
    print(f"Interaction (difference-in-differences):")
    print(f"  Mean = {mean_int:+.2f} ± {std_int:.2f} (n={len(results)})")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} at α=0.05")
    print(f"{'='*60}")

    # Also print the marginal effects
    delta_exps = [r["delta_exp"] for r in results]
    delta_bm25s = [r["delta_bm25"] for r in results]
    print(f"\nExpanded pool delta:  mean={np.mean(delta_exps):+.2f} ± {np.std(delta_exps, ddof=1):.2f}")
    print(f"BM25 pool delta:     mean={np.mean(delta_bm25s):+.2f} ± {np.std(delta_bm25s, ddof=1):.2f}")

    # LaTeX table
    print(f"\n=== LaTeX (for paper) ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Seed & \multicolumn{2}{c}{Expanded pool} & \multicolumn{2}{c}{BM25-only pool} \\")
    print(r"     & Graph & BM25 & Graph & BM25 \\")
    print(r"\midrule")
    for r in results:
        print(f"{r['seed']:2d} & {r['g_exp']:.2f} & {r['b_exp']:.2f} & {r['g_bm25']:.2f} & {r['b_bm25']:.2f} \\\\")
    print(r"\midrule")
    print(f"Mean & {np.mean([r['g_exp'] for r in results]):.2f} & {np.mean([r['b_exp'] for r in results]):.2f} "
          f"& {np.mean([r['g_bm25'] for r in results]):.2f} & {np.mean([r['b_bm25'] for r in results]):.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
