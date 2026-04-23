#!/usr/bin/env python3
"""Conditional reranking accuracy analysis for the CodeGRIP paper.

Decomposes R@k into:
    R@k = Oracle_rate * Conditional_R@k

where:
  - Oracle_rate: fraction of examples where ANY ground-truth file is in the
    candidate pool (measures coverage / retrieval quality).
  - Conditional R@k: R@k computed ONLY among examples where at least one GT
    file is in the pool (measures pure reranking quality).

This separates the coverage gain (getting GT into the pool) from the ranking
gain (ranking GT higher once it is in the pool).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/chenlibin/grepo_agent")

# ---------------------------------------------------------------------------
# Prediction files: each line has {repo, issue_id, ground_truth, predicted,
#   gt_in_candidates, metrics: {hit@k, acc@k, recall@k for k in 1,3,5,10,20}}
# ---------------------------------------------------------------------------
CONDITIONS = [
    {
        "name": "BM25 pool + BM25-only",
        "pred_path": ROOT / "experiments/rankft_runA_bm25only/eval_bm25pool/predictions.jsonl",
    },
    {
        "name": "BM25 pool + Graph-hard",
        "pred_path": ROOT / "experiments/rankft_runB_graph/eval_bm25pool/predictions.jsonl",
    },
    {
        "name": "Expanded  + BM25-only",
        "pred_path": ROOT / "experiments/rankft_runA_bm25only/eval_merged_rerank/predictions.jsonl",
    },
    {
        "name": "Expanded  + Graph-hard",
        "pred_path": ROOT / "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
    },
]

KS = [1, 5, 10]


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_conditional_metrics(preds: list[dict], ks: list[int] = KS) -> dict:
    """Compute oracle, conditional, and unconditional hit@k / recall@k."""
    total = len(preds)
    gt_in_count = sum(1 for p in preds if p["gt_in_candidates"])

    oracle_rate = gt_in_count / total if total > 0 else 0.0

    # --- Unconditional metrics (over ALL examples) ---
    uncond = {}
    for k in ks:
        key = f"hit@{k}"
        uncond[key] = sum(p["metrics"][key] for p in preds) / total if total else 0.0

    # --- Conditional metrics (only examples where GT is in pool) ---
    cond = {}
    cond_preds = [p for p in preds if p["gt_in_candidates"]]
    n_cond = len(cond_preds)
    for k in ks:
        key = f"hit@{k}"
        cond[key] = sum(p["metrics"][key] for p in cond_preds) / n_cond if n_cond else 0.0

    # --- Also compute acc@k (all GT files in top-k) conditional ---
    uncond_acc = {}
    cond_acc = {}
    for k in ks:
        key = f"acc@{k}"
        uncond_acc[key] = sum(p["metrics"][key] for p in preds) / total if total else 0.0
        cond_acc[key] = sum(p["metrics"][key] for p in cond_preds) / n_cond if n_cond else 0.0

    return {
        "total": total,
        "gt_in_count": gt_in_count,
        "oracle_rate": oracle_rate,
        "uncond_hit": uncond,
        "cond_hit": cond,
        "uncond_acc": uncond_acc,
        "cond_acc": cond_acc,
    }


def main():
    results = []
    for cond in CONDITIONS:
        path = cond["pred_path"]
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        preds = load_predictions(path)
        metrics = compute_conditional_metrics(preds)
        metrics["name"] = cond["name"]
        results.append(metrics)

    # -----------------------------------------------------------------------
    # Print decomposition table
    # -----------------------------------------------------------------------
    print("=" * 110)
    print("Conditional Reranking Accuracy Decomposition")
    print("  R@k(unconditional) = Oracle_rate x R@k(conditional)")
    print("=" * 110)

    # Header
    header_parts = [f"{'Condition':<27}", f"{'N':>5}", f"{'Oracle':>7}"]
    for k in KS:
        header_parts.append(f"{'Cond R@'+str(k):>9}")
        header_parts.append(f"{'R@'+str(k):>7}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for r in results:
        parts = [
            f"{r['name']:<27}",
            f"{r['total']:>5}",
            f"{r['oracle_rate']*100:>6.1f}%",
        ]
        for k in KS:
            key = f"hit@{k}"
            parts.append(f"{r['cond_hit'][key]*100:>8.2f}%")
            parts.append(f"{r['uncond_hit'][key]*100:>6.2f}%")
        print(" | ".join(parts))

    print()

    # -----------------------------------------------------------------------
    # Delta table: show gains from coverage vs reranking
    # -----------------------------------------------------------------------
    print("=" * 110)
    print("Gain Decomposition (relative to BM25 pool + BM25-only baseline)")
    print("=" * 110)

    if len(results) < 4:
        print("Not enough conditions to compute deltas.")
        return

    base = results[0]  # BM25 pool + BM25-only

    print(f"{'Comparison':<42} | {'Total dR@1':>10} | {'Coverage':>10} | {'Ranking':>10} | {'Interaction':>12}")
    print("-" * 95)

    for r in results[1:]:
        # Decompose: R@1 = Oracle * CondR@1
        # dR@1 = (O_new * C_new) - (O_base * C_base)
        #       = dO * C_base + O_base * dC + dO * dC
        # where dO = O_new - O_base, dC = C_new - C_base
        o_base = base["oracle_rate"]
        c_base = base["cond_hit"]["hit@1"]
        o_new = r["oracle_rate"]
        c_new = r["cond_hit"]["hit@1"]

        d_oracle = o_new - o_base
        d_cond = c_new - c_base
        total_delta = (o_new * c_new) - (o_base * c_base)

        coverage_term = d_oracle * c_base  # coverage gain at baseline ranking
        ranking_term = o_base * d_cond     # ranking gain at baseline coverage
        interaction = d_oracle * d_cond    # interaction (both change)

        label = f"{base['name']} -> {r['name']}"
        print(
            f"{label:<42} | {total_delta*100:>+9.2f}% | "
            f"{coverage_term*100:>+9.2f}% | {ranking_term*100:>+9.2f}% | "
            f"{interaction*100:>+11.2f}%"
        )

    print()

    # -----------------------------------------------------------------------
    # Acc@k table (all GT in top-k, conditional)
    # -----------------------------------------------------------------------
    print("=" * 110)
    print("Conditional Acc@k (ALL GT files in top-k, only among examples where GT is in pool)")
    print("=" * 110)

    header_parts = [f"{'Condition':<27}", f"{'Oracle':>7}"]
    for k in KS:
        header_parts.append(f"{'Cond A@'+str(k):>9}")
        header_parts.append(f"{'A@'+str(k):>7}")
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for r in results:
        parts = [
            f"{r['name']:<27}",
            f"{r['oracle_rate']*100:>6.1f}%",
        ]
        for k in KS:
            key = f"acc@{k}"
            parts.append(f"{r['cond_acc'][key]*100:>8.2f}%")
            parts.append(f"{r['uncond_acc'][key]*100:>6.2f}%")
        print(" | ".join(parts))

    print()

    # -----------------------------------------------------------------------
    # Sanity check: unconditional = oracle * conditional (should hold exactly)
    # -----------------------------------------------------------------------
    print("Sanity check (unconditional ~= oracle * conditional):")
    all_ok = True
    for r in results:
        for k in KS:
            key = f"hit@{k}"
            product = r["oracle_rate"] * r["cond_hit"][key] * 100
            actual = r["uncond_hit"][key] * 100
            if abs(product - actual) > 0.01:
                print(f"  MISMATCH {r['name']} R@{k}: product={product:.4f} actual={actual:.4f}")
                all_ok = False
    if all_ok:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
