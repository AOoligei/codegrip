#!/usr/bin/env python3
"""
Collect seed robustness results for rankft_runB_graph and rankft_runA_bm25only.
Computes mean ± std across seeds and paired t-test.
Usage: python scripts/collect_seed_results.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path("experiments")

SEEDS = [42, 1, 2, 3, 4]  # seed 42 = original runs

GRAPH_EXPS = {
    42: "rankft_runB_graph",
    1:  "rankft_runB_graph_seed1",
    2:  "rankft_runB_graph_seed2",
    3:  "rankft_runB_graph_seed3",
    4:  "rankft_runB_graph_seed4",
}
BM25_EXPS = {
    42: "rankft_runA_bm25only",
    1:  "rankft_runA_bm25only_seed1",
    2:  "rankft_runA_bm25only_seed2",
    3:  "rankft_runA_bm25only_seed3",
    4:  "rankft_runA_bm25only_seed4",
}


def load_r1(exp_name):
    p = BASE / exp_name / "eval_merged_rerank" / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["overall"]
    return d["recall@1"]


graph_r1 = {}
bm25_r1 = {}
for seed in SEEDS:
    g = load_r1(GRAPH_EXPS[seed])
    b = load_r1(BM25_EXPS[seed])
    if g is not None:
        graph_r1[seed] = g
    if b is not None:
        bm25_r1[seed] = b

print("=== Seed Robustness Results ===\n")
print(f"{'Seed':<8} {'Graph R@1':>10} {'BM25 R@1':>10} {'Delta':>8}")
print("-" * 40)
deltas = []
for seed in SEEDS:
    g = graph_r1.get(seed)
    b = bm25_r1.get(seed)
    if g is None and b is None:
        print(f"{seed:<8} {'MISSING':>10} {'MISSING':>10}")
        continue
    delta = (g - b) if (g and b) else None
    g_s = f"{g:.2f}" if g else "MISSING"
    b_s = f"{b:.2f}" if b else "MISSING"
    d_s = f"{delta:+.2f}" if delta is not None else "N/A"
    print(f"{seed:<8} {g_s:>10} {b_s:>10} {d_s:>8}")
    if delta is not None:
        deltas.append(delta)

available_seeds = [s for s in SEEDS if s in graph_r1 and s in bm25_r1]
if len(available_seeds) >= 2:
    g_vals = [graph_r1[s] for s in available_seeds]
    b_vals = [bm25_r1[s] for s in available_seeds]

    print(f"\nGraph:  mean={np.mean(g_vals):.2f} ± {np.std(g_vals, ddof=1):.2f}")
    print(f"BM25:   mean={np.mean(b_vals):.2f} ± {np.std(b_vals, ddof=1):.2f}")
    print(f"Delta:  mean={np.mean(deltas):.2f} ± {np.std(deltas, ddof=1):.2f}")

    if len(available_seeds) >= 3:
        t, p = stats.ttest_rel(g_vals, b_vals)
        print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
    else:
        print(f"(Need 3 seeds for paired t-test; have {len(available_seeds)})")

    print("\n=== LaTeX (for paper) ===")
    g_mean, g_std = np.mean(g_vals), np.std(g_vals, ddof=1)
    b_mean, b_std = np.mean(b_vals), np.std(b_vals, ddof=1)
    print(f"Graph-hard & {g_mean:.2f} $\\pm$ {g_std:.2f} \\\\")
    print(f"No-graph (BM25+random) & {b_mean:.2f} $\\pm$ {b_std:.2f} \\\\")
else:
    print(f"\nOnly {len(available_seeds)} seed(s) complete — run more seeds.")
