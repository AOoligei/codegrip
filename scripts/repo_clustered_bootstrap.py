"""Repo-clustered bootstrap significance testing.

Instead of resampling individual examples (which assumes independence),
resample whole repositories to account for within-repo dependence.
"""

import json
import numpy as np
from collections import defaultdict
import argparse


def load_predictions(path):
    """Load predictions and group by repo."""
    repo_examples = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            repo = d["repo"]
            repo_examples[repo].append(d)
    return dict(repo_examples)


def compute_recall_at_k(examples, k=1):
    """Compute mean recall@k from a list of examples."""
    scores = []
    for ex in examples:
        gt = set(ex["ground_truth"])
        pred = ex["predicted"][:k]
        if len(gt) == 0:
            continue
        recall = len(set(pred) & gt) / len(gt)
        scores.append(recall)
    return np.mean(scores) if scores else 0.0


def repo_clustered_bootstrap(repo_examples_a, repo_examples_b,
                              metric_fn, n_boot=10000, seed=42):
    """Cluster bootstrap: resample repos with replacement.

    Args:
        repo_examples_a: dict[repo -> list[examples]] for system A
        repo_examples_b: dict[repo -> list[examples]] for system B
        metric_fn: function(examples) -> float
        n_boot: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with mean_a, mean_b, delta, ci_lower, ci_upper, p_value
    """
    rng = np.random.RandomState(seed)
    repos = sorted(set(repo_examples_a.keys()) & set(repo_examples_b.keys()))
    n_repos = len(repos)

    # Observed statistics
    all_a = [ex for r in repos for ex in repo_examples_a[r]]
    all_b = [ex for r in repos for ex in repo_examples_b[r]]
    obs_a = metric_fn(all_a)
    obs_b = metric_fn(all_b)
    obs_delta = obs_b - obs_a

    # Bootstrap
    boot_deltas = []
    for _ in range(n_boot):
        # Sample repos with replacement
        sampled_repos = rng.choice(repos, size=n_repos, replace=True)
        boot_a = [ex for r in sampled_repos for ex in repo_examples_a[r]]
        boot_b = [ex for r in sampled_repos for ex in repo_examples_b[r]]
        delta = metric_fn(boot_b) - metric_fn(boot_a)
        boot_deltas.append(delta)

    boot_deltas = np.array(boot_deltas)

    # Confidence interval
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    # p-value: fraction of bootstrap samples where delta <= 0
    p_value = np.mean(boot_deltas <= 0)

    return {
        "mean_a": obs_a,
        "mean_b": obs_b,
        "delta": obs_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_repos": n_repos,
        "n_examples_a": len(all_a),
        "n_examples_b": len(all_b),
        "boot_std": np.std(boot_deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_a", required=True, help="Predictions file for system A (baseline)")
    parser.add_argument("--pred_b", required=True, help="Predictions file for system B (treatment)")
    parser.add_argument("--k", type=int, default=1, help="Recall@k")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading predictions...")
    repo_a = load_predictions(args.pred_a)
    repo_b = load_predictions(args.pred_b)

    metric_fn = lambda examples: compute_recall_at_k(examples, k=args.k)

    print(f"Running repo-clustered bootstrap (n_boot={args.n_boot}, k={args.k})...")
    result = repo_clustered_bootstrap(repo_a, repo_b, metric_fn,
                                       n_boot=args.n_boot, seed=args.seed)

    print(f"\n=== Repo-Clustered Bootstrap (R@{args.k}) ===")
    print(f"System A: {result['mean_a']*100:.2f}% ({result['n_examples_a']} examples)")
    print(f"System B: {result['mean_b']*100:.2f}% ({result['n_examples_b']} examples)")
    print(f"Delta:    {result['delta']*100:+.2f}pp")
    print(f"95% CI:   [{result['ci_lower']*100:+.2f}, {result['ci_upper']*100:+.2f}]pp")
    print(f"Boot std: {result['boot_std']*100:.2f}pp")
    print(f"p-value:  {result['p_value']:.4f}")
    print(f"Repos:    {result['n_repos']}")
    print()

    # Also run per-example bootstrap for comparison
    all_a = [ex for r in sorted(repo_a) for ex in repo_a[r]]
    all_b = [ex for r in sorted(repo_b) for ex in repo_b[r]]

    rng = np.random.RandomState(args.seed)
    n = min(len(all_a), len(all_b))
    scores_a = np.array([len(set(ex["predicted"][:args.k]) & set(ex["ground_truth"])) / max(len(ex["ground_truth"]), 1) for ex in all_a[:n]])
    scores_b = np.array([len(set(ex["predicted"][:args.k]) & set(ex["ground_truth"])) / max(len(ex["ground_truth"]), 1) for ex in all_b[:n]])

    boot_deltas_iid = []
    for _ in range(args.n_boot):
        idx = rng.randint(0, n, size=n)
        delta = np.mean(scores_b[idx]) - np.mean(scores_a[idx])
        boot_deltas_iid.append(delta)
    boot_deltas_iid = np.array(boot_deltas_iid)

    print(f"=== Per-Example Bootstrap (R@{args.k}, for comparison) ===")
    print(f"Delta:    {(np.mean(scores_b) - np.mean(scores_a))*100:+.2f}pp")
    print(f"95% CI:   [{np.percentile(boot_deltas_iid, 2.5)*100:+.2f}, {np.percentile(boot_deltas_iid, 97.5)*100:+.2f}]pp")
    print(f"Boot std: {np.std(boot_deltas_iid)*100:.2f}pp")
    print(f"p-value:  {np.mean(boot_deltas_iid <= 0):.4f}")


if __name__ == "__main__":
    main()
