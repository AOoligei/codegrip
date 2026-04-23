#!/usr/bin/env python3
"""Confidence-gated path->code routing evaluation for CodeGRIP.

When the path-only model's score gap (top1 - top2) is small (low confidence),
fall back to the code-residual model's ranking. Otherwise use path-only.

Workflow:
  1. Sweep threshold on val set to find optimal gap threshold.
  2. Apply best threshold to test set and report results by confidence slice.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
VAL_PATH_ONLY = Path("/data/chenlibin/grepo_agent_experiments/val_eval/path_only/predictions.jsonl")
VAL_CODE_RES = Path("/data/chenlibin/grepo_agent_experiments/val_eval/code_residual_v2/predictions.jsonl")
TEST_PATH_ONLY = Path("/data/chenlibin/grepo_agent_experiments/multiseed_seed42/eval_graph_rerank/predictions.jsonl")
TEST_CODE_RES = Path("/data/chenlibin/grepo_agent_experiments/code_residual_7b_v2/eval_graph/predictions.jsonl")


def load_predictions(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_hit_at_k(predicted: list[str], gt: set, k: int) -> float:
    """Partial recall@k: fraction of GT files present in top-k."""
    if not gt:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gt) / len(gt)


def score_gap(scores: list[float]) -> float:
    """Confidence = score gap between rank-1 and rank-2 candidates."""
    if len(scores) < 2:
        return float("inf")
    return scores[0] - scores[1]


def gated_routing(
    path_preds: list[dict],
    code_preds: list[dict],
    threshold: float,
) -> list[dict]:
    """Apply confidence-gated routing.

    If path-only gap > threshold: use path-only ranking.
    Else: use code-residual ranking.

    Returns list of dicts with keys: repo, issue_id, ground_truth,
    predicted, source (str: 'path' or 'code'), gap.
    """
    results = []
    for pp, cp in zip(path_preds, code_preds):
        assert pp["repo"] == cp["repo"] and pp["issue_id"] == cp["issue_id"]
        gap = score_gap(pp["scores"])
        if gap > threshold:
            predicted = pp["predicted"]
            source = "path"
        else:
            predicted = cp["predicted"]
            source = "code"
        results.append({
            "repo": pp["repo"],
            "issue_id": pp["issue_id"],
            "ground_truth": pp["ground_truth"],
            "predicted": predicted,
            "source": source,
            "gap": gap,
            "gt_in_candidates": pp["gt_in_candidates"],
        })
    return results


def evaluate_set(results: list[dict], ks=(1, 3, 5, 10)) -> dict:
    """Compute hit@k for a set of gated predictions."""
    metrics = {}
    for k in ks:
        hits = [
            compute_hit_at_k(r["predicted"], set(r["ground_truth"]), k)
            for r in results
        ]
        metrics[f"hit@{k}"] = np.mean(hits) * 100
    metrics["n"] = len(results)
    # Fraction routed to code
    n_code = sum(1 for r in results if r["source"] == "code")
    metrics["code_frac"] = n_code / len(results) * 100 if results else 0.0
    return metrics


def sweep_threshold(
    path_preds: list[dict],
    code_preds: list[dict],
    n_steps: int = 200,
) -> tuple[float, list[tuple[float, dict]]]:
    """Sweep threshold on val set. Returns (best_threshold, curve)."""
    # Collect all gaps to define sweep range
    gaps = [score_gap(p["scores"]) for p in path_preds]
    lo, hi = np.percentile(gaps, 1), np.percentile(gaps, 99)
    thresholds = np.linspace(lo, hi, n_steps)

    curve = []
    best_hit1 = -1.0
    best_thresh = 0.0

    for t in thresholds:
        results = gated_routing(path_preds, code_preds, t)
        m = evaluate_set(results)
        curve.append((float(t), m))
        if m["hit@1"] > best_hit1:
            best_hit1 = m["hit@1"]
            best_thresh = float(t)

    return best_thresh, curve


def print_table(label: str, metrics: dict):
    """Pretty print metrics."""
    ks_str = "  ".join(f"hit@{k}: {metrics.get(f'hit@{k}', 0):.2f}" for k in (1, 3, 5, 10))
    print(f"  {label:30s}  n={metrics['n']:5d}  {ks_str}  code%={metrics['code_frac']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Confidence-gated routing eval")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: stdout only)")
    args = parser.parse_args()

    print("=" * 90)
    print("Confidence-Gated Path->Code Routing Evaluation")
    print("=" * 90)

    # ── Load predictions ───────────────────────────────────────────────────
    print("\nLoading predictions...")
    val_path = load_predictions(VAL_PATH_ONLY)
    val_code = load_predictions(VAL_CODE_RES)
    test_path = load_predictions(TEST_PATH_ONLY)
    test_code = load_predictions(TEST_CODE_RES)
    print(f"  Val: {len(val_path)} examples, Test: {len(test_path)} examples")

    # ── Baselines ──────────────────────────────────────────────────────────
    print("\n--- Baselines (Test Set) ---")
    path_only_m = evaluate_set([
        {"predicted": p["predicted"], "ground_truth": p["ground_truth"],
         "source": "path", "gap": 0, "gt_in_candidates": p["gt_in_candidates"]}
        for p in test_path
    ])
    code_only_m = evaluate_set([
        {"predicted": p["predicted"], "ground_truth": p["ground_truth"],
         "source": "code", "gap": 0, "gt_in_candidates": p["gt_in_candidates"]}
        for p in test_code
    ])
    print_table("Path-only (all)", path_only_m)
    print_table("Code-residual (all)", code_only_m)

    # ── Threshold sweep on val ─────────────────────────────────────────────
    print("\n--- Threshold Sweep (Val Set) ---")
    best_thresh, curve = sweep_threshold(val_path, val_code)
    print(f"  Best threshold: {best_thresh:.4f}")

    # Show a few points around the optimum
    sorted_curve = sorted(curve, key=lambda x: -x[1]["hit@1"])
    print("  Top-5 thresholds by val hit@1:")
    for t, m in sorted_curve[:5]:
        print(f"    thresh={t:.4f}  hit@1={m['hit@1']:.2f}  code%={m['code_frac']:.1f}")

    # ── Apply best threshold to test ───────────────────────────────────────
    print(f"\n--- Test Results with threshold={best_thresh:.4f} ---")
    test_gated = gated_routing(test_path, test_code, best_thresh)
    gated_m = evaluate_set(test_gated)

    print_table("Gated routing (overall)", gated_m)
    delta = gated_m["hit@1"] - path_only_m["hit@1"]
    print(f"  Delta vs path-only: {delta:+.2f} pp")

    # ── Confidence slices ──────────────────────────────────────────────────
    print("\n--- Confidence Slice Analysis (Test Set) ---")
    test_gaps = [score_gap(p["scores"]) for p in test_path]
    median_gap = np.median(test_gaps)
    print(f"  Median gap: {median_gap:.4f}, Mean gap: {np.mean(test_gaps):.4f}")

    # High-confidence slice: gap >= median
    high_idx = [i for i, g in enumerate(test_gaps) if g >= median_gap]
    low_idx = [i for i, g in enumerate(test_gaps) if g < median_gap]

    # Path-only on slices
    high_path = evaluate_set([
        {"predicted": test_path[i]["predicted"],
         "ground_truth": test_path[i]["ground_truth"],
         "source": "path", "gap": 0, "gt_in_candidates": test_path[i]["gt_in_candidates"]}
        for i in high_idx
    ])
    low_path = evaluate_set([
        {"predicted": test_path[i]["predicted"],
         "ground_truth": test_path[i]["ground_truth"],
         "source": "path", "gap": 0, "gt_in_candidates": test_path[i]["gt_in_candidates"]}
        for i in low_idx
    ])

    # Gated on slices
    high_gated = evaluate_set([test_gated[i] for i in high_idx])
    low_gated = evaluate_set([test_gated[i] for i in low_idx])

    # Code-only on slices
    high_code = evaluate_set([
        {"predicted": test_code[i]["predicted"],
         "ground_truth": test_code[i]["ground_truth"],
         "source": "code", "gap": 0, "gt_in_candidates": test_code[i]["gt_in_candidates"]}
        for i in high_idx
    ])
    low_code = evaluate_set([
        {"predicted": test_code[i]["predicted"],
         "ground_truth": test_code[i]["ground_truth"],
         "source": "code", "gap": 0, "gt_in_candidates": test_code[i]["gt_in_candidates"]}
        for i in low_idx
    ])

    print("\n  High-confidence (gap >= median):")
    print_table("  Path-only", high_path)
    print_table("  Code-residual", high_code)
    print_table("  Gated", high_gated)
    delta_high = high_gated["hit@1"] - high_path["hit@1"]
    print(f"    Delta: {delta_high:+.2f} pp")

    print("\n  Low-confidence (gap < median):")
    print_table("  Path-only", low_path)
    print_table("  Code-residual", low_code)
    print_table("  Gated", low_gated)
    delta_low = low_gated["hit@1"] - low_path["hit@1"]
    print(f"    Delta: {delta_low:+.2f} pp")

    # ── Routing breakdown ──────────────────────────────────────────────────
    print("\n--- Routing Breakdown (Test Set) ---")
    n_path = sum(1 for r in test_gated if r["source"] == "path")
    n_code = sum(1 for r in test_gated if r["source"] == "code")
    print(f"  Routed to path: {n_path} ({n_path/len(test_gated)*100:.1f}%)")
    print(f"  Routed to code: {n_code} ({n_code/len(test_gated)*100:.1f}%)")

    # Among those routed to code, what's the accuracy improvement?
    code_routed = [r for r in test_gated if r["source"] == "code"]
    code_routed_path = []
    for r in code_routed:
        # Find matching path prediction
        for p in test_path:
            if p["repo"] == r["repo"] and p["issue_id"] == r["issue_id"]:
                code_routed_path.append({
                    "predicted": p["predicted"],
                    "ground_truth": p["ground_truth"],
                    "source": "path", "gap": 0,
                    "gt_in_candidates": p["gt_in_candidates"],
                })
                break

    if code_routed_path:
        m_code_sub_path = evaluate_set(code_routed_path)
        m_code_sub_code = evaluate_set(code_routed)
        print(f"\n  Among code-routed examples (n={len(code_routed)}):")
        print_table("  Would-be path-only", m_code_sub_path)
        print_table("  Actual (code-res)", m_code_sub_code)
        delta_sub = m_code_sub_code["hit@1"] - m_code_sub_path["hit@1"]
        print(f"    Delta: {delta_sub:+.2f} pp")

    # ── Score fusion variant ──────────────────────────────────────────────
    print("\n--- Score Fusion Variant (Test Set) ---")
    print("  Fusing path + code scores with weight alpha for code:")

    # Build lookup for code scores by (repo, issue_id, filename)
    def build_score_map(preds):
        """Map (repo, issue_id) -> {filename: score}."""
        m = {}
        for p in preds:
            key = (p["repo"], p["issue_id"])
            m[key] = dict(zip(p["predicted"], p["scores"]))
        return m

    test_code_scores = build_score_map(test_code)
    test_path_scores = build_score_map(test_path)
    val_code_scores = build_score_map(val_code)
    val_path_scores = build_score_map(val_path)

    def fused_routing(path_preds, path_scores_map, code_scores_map,
                      threshold, alpha):
        """Confidence-gated with score fusion for low-confidence cases.

        High confidence (gap > threshold): use path ranking.
        Low confidence (gap <= threshold): re-rank by (1-alpha)*path + alpha*code.
        """
        results = []
        for pp in path_preds:
            key = (pp["repo"], pp["issue_id"])
            gap = score_gap(pp["scores"])
            if gap > threshold:
                predicted = pp["predicted"]
                source = "path"
            else:
                # Fuse scores for all candidates in path prediction
                p_scores = path_scores_map[key]
                c_scores = code_scores_map.get(key, {})
                # Use all path candidates, fuse with code scores where available
                candidates = pp["predicted"]
                fused = []
                for fname in candidates:
                    ps = p_scores.get(fname, -999)
                    cs = c_scores.get(fname, -999)
                    # Normalize: z-score within each example
                    fused.append((fname, (1 - alpha) * ps + alpha * cs))
                fused.sort(key=lambda x: -x[1])
                predicted = [f[0] for f in fused]
                source = "fused"
            results.append({
                "predicted": predicted,
                "ground_truth": pp["ground_truth"],
                "source": source,
                "gap": gap,
                "gt_in_candidates": pp["gt_in_candidates"],
                "repo": pp["repo"],
                "issue_id": pp["issue_id"],
            })
        return results

    # Sweep alpha and threshold jointly on val
    best_val_hit1 = -1.0
    best_alpha = 0.0
    best_fusion_thresh = 0.0

    for alpha in np.arange(0.1, 0.6, 0.05):
        val_gaps = [score_gap(p["scores"]) for p in val_path]
        for pct in np.arange(5, 60, 5):
            thresh = np.percentile(val_gaps, pct)
            results = fused_routing(val_path, val_path_scores, val_code_scores,
                                    thresh, alpha)
            m = evaluate_set(results)
            if m["hit@1"] > best_val_hit1:
                best_val_hit1 = m["hit@1"]
                best_alpha = float(alpha)
                best_fusion_thresh = float(thresh)

    print(f"  Best alpha={best_alpha:.2f}, threshold={best_fusion_thresh:.4f} "
          f"(val hit@1={best_val_hit1:.2f})")

    # Apply to test
    test_fused = fused_routing(test_path, test_path_scores, test_code_scores,
                               best_fusion_thresh, best_alpha)
    fused_m = evaluate_set(test_fused)
    print_table("Fused routing (overall)", fused_m)
    delta_fused = fused_m["hit@1"] - path_only_m["hit@1"]
    print(f"  Delta vs path-only: {delta_fused:+.2f} pp")

    # Slice analysis for fusion
    fused_high = evaluate_set([test_fused[i] for i in high_idx])
    fused_low = evaluate_set([test_fused[i] for i in low_idx])
    print("\n  High-confidence:")
    print_table("  Fused", fused_high)
    print(f"    Delta vs path: {fused_high['hit@1'] - high_path['hit@1']:+.2f} pp")
    print("  Low-confidence:")
    print_table("  Fused", fused_low)
    print(f"    Delta vs path: {fused_low['hit@1'] - low_path['hit@1']:+.2f} pp")

    # ── Bootstrap confidence interval ──────────────────────────────────────
    print("\n--- Bootstrap 95% CI for gated hit@1 improvement ---")
    n_boot = 10000
    path_hits = np.array([
        compute_hit_at_k(p["predicted"], set(p["ground_truth"]), 1)
        for p in test_path
    ])
    gated_hits = np.array([
        compute_hit_at_k(r["predicted"], set(r["ground_truth"]), 1)
        for r in test_gated
    ])
    deltas_boot = []
    n = len(path_hits)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        d = gated_hits[idx].mean() - path_hits[idx].mean()
        deltas_boot.append(d)
    deltas_boot = np.array(deltas_boot) * 100  # to pp
    ci_lo, ci_hi = np.percentile(deltas_boot, [2.5, 97.5])
    print(f"  Delta hit@1: {delta:.2f} pp, 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

    # ── Save results ───────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save threshold curve
        curve_data = [{"threshold": t, **m} for t, m in curve]
        with open(out_dir / "threshold_curve.jsonl", "w") as f:
            for d in curve_data:
                f.write(json.dumps(d) + "\n")

        # Save summary
        summary = {
            "best_threshold": best_thresh,
            "val_best_hit1": sorted_curve[0][1]["hit@1"],
            "test_path_only_hit1": path_only_m["hit@1"],
            "test_code_only_hit1": code_only_m["hit@1"],
            "test_gated_hit1": gated_m["hit@1"],
            "test_delta_pp": delta,
            "test_delta_ci95": [ci_lo, ci_hi],
            "n_routed_path": n_path,
            "n_routed_code": n_code,
            "median_gap": float(median_gap),
            "high_conf_path_hit1": high_path["hit@1"],
            "high_conf_gated_hit1": high_gated["hit@1"],
            "low_conf_path_hit1": low_path["hit@1"],
            "low_conf_gated_hit1": low_gated["hit@1"],
            "fusion_alpha": best_alpha,
            "fusion_threshold": best_fusion_thresh,
            "fusion_val_hit1": best_val_hit1,
            "fusion_test_hit1": fused_m["hit@1"],
            "fusion_delta_pp": delta_fused,
            "fusion_high_conf_hit1": fused_high["hit@1"],
            "fusion_low_conf_hit1": fused_low["hit@1"],
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save per-example gated predictions
        with open(out_dir / "gated_predictions.jsonl", "w") as f:
            for r in test_gated:
                f.write(json.dumps(r) + "\n")

        print(f"\n  Results saved to {out_dir}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
