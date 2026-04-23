"""
Paired bootstrap significance test comparing two models' per-example predictions.

Usage:
    python scripts/bootstrap_significance.py \
        --pred_a <dir_a>/predictions.jsonl \
        --pred_b <dir_b>/predictions.jsonl

Deterministic (seed 42), CPU-only, standard libraries + numpy.
"""

import argparse
import json
import numpy as np


def load_predictions(path):
    """Load predictions.jsonl, return dict keyed by (repo, issue_id_str)."""
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # Normalize issue_id to str for consistent keying
            key = (rec["repo"], str(rec["issue_id"]))
            data[key] = rec
    return data


def compute_hit_at_1(rec):
    """Compute hit@1: 1 if any ground truth file is in top-1 predicted."""
    gt = set(rec["ground_truth"])
    predicted = rec["predicted"]
    if len(predicted) == 0:
        return 0.0
    return 1.0 if predicted[0] in gt else 0.0


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test")
    parser.add_argument("--pred_a", required=True, help="Path to model A predictions.jsonl")
    parser.add_argument("--pred_b", required=True, help="Path to model B predictions.jsonl")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--metric", default="hit_at_1", choices=["hit_at_1"],
                        help="Metric to compare")
    args = parser.parse_args()

    np.random.seed(42)

    pred_a = load_predictions(args.pred_a)
    pred_b = load_predictions(args.pred_b)

    # Align by common keys
    common_keys = sorted(set(pred_a.keys()) & set(pred_b.keys()))
    n = len(common_keys)

    if n == 0:
        print("ERROR: No common examples found between the two prediction files.")
        return

    print(f"Aligned examples: {n}")
    print(f"Model A only: {len(pred_a) - n}")
    print(f"Model B only: {len(pred_b) - n}")
    print()

    # Compute per-example metrics
    scores_a = np.array([compute_hit_at_1(pred_a[k]) for k in common_keys])
    scores_b = np.array([compute_hit_at_1(pred_b[k]) for k in common_keys])
    diffs = scores_b - scores_a  # positive means B is better

    observed_mean_diff = np.mean(diffs)
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    # Bootstrap
    boot_diffs = np.empty(args.n_bootstrap)
    indices = np.arange(n)
    for i in range(args.n_bootstrap):
        sample = np.random.choice(indices, size=n, replace=True)
        boot_diffs[i] = np.mean(diffs[sample])

    # 95% CI
    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)

    # Two-sided p-value: fraction of bootstrap samples where sign flips
    # (or is zero) relative to observed direction
    if observed_mean_diff >= 0:
        p_value = np.mean(boot_diffs <= 0)
    else:
        p_value = np.mean(boot_diffs >= 0)
    # Two-sided
    p_value = min(2 * p_value, 1.0)

    # Report
    print(f"{'Metric':<20s}: {args.metric}")
    print(f"{'Model A mean':<20s}: {mean_a:.4f}")
    print(f"{'Model B mean':<20s}: {mean_b:.4f}")
    print(f"{'Mean diff (B - A)':<20s}: {observed_mean_diff:+.4f}")
    print(f"{'95% CI':<20s}: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"{'p-value (two-sided)':<20s}: {p_value:.4f}")
    print(f"{'N examples':<20s}: {n}")
    print(f"{'N bootstrap':<20s}: {args.n_bootstrap}")

    if p_value < 0.05:
        print("\nResult: SIGNIFICANT at alpha=0.05")
    else:
        print("\nResult: NOT significant at alpha=0.05")


if __name__ == "__main__":
    main()
