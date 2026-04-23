"""
Full analysis of all experiments: comparison table, ablation, per-repo analysis.
Run after all experiments are evaluated.

Usage:
    python scripts/full_analysis.py
"""

import json
import os
from collections import defaultdict

EXPERIMENTS_DIR = "experiments"
GAT_RESULTS = {"hit@1": 14.80, "hit@5": 31.51, "hit@10": 37.40, "hit@20": 41.25}
AGENTLESS_RESULTS = {"hit@1": 13.65, "hit@5": 21.86, "hit@10": 23.43, "hit@20": 23.43}


def load_summary(exp_name, stage="eval_reranked"):
    """Load summary.json from an experiment stage."""
    path = os.path.join(EXPERIMENTS_DIR, exp_name, stage, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_comparison_table():
    """Print comprehensive comparison table."""
    experiments = [
        ("exp1_sft_only", "SFT (Qwen2.5-7B)"),
        ("exp5_coder_sft_only", "SFT (Coder-7B)"),
        ("exp6_warmstart_cochange", "Warm-start CC"),
        ("exp7_multitask_sft", "Multi-task SFT"),
        ("ensemble_rrf", "Ensemble (RRF)"),
    ]

    stages = [
        ("eval_filetree", "base"),
        ("eval_unified_expansion", "+expand"),
        ("eval_reranked", "+rerank"),
    ]

    print("=" * 85)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 85)
    print(f"{'Method':<35} {'H@1':>7} {'H@5':>7} {'H@10':>7} {'H@20':>7}")
    print("-" * 85)

    # Baselines
    for name, results in [("GAT (GREPO paper)", GAT_RESULTS), ("Agentless (paper)", AGENTLESS_RESULTS)]:
        print(f"{name:<35} {results['hit@1']:7.2f} {results['hit@5']:7.2f} "
              f"{results['hit@10']:7.2f} {results['hit@20']:7.2f}")

    print("-" * 85)

    # Our experiments
    best_result = None
    for exp_name, display_name in experiments:
        for stage_dir, stage_label in stages:
            summary = load_summary(exp_name, stage_dir)
            if summary is None:
                continue

            o = summary['overall']
            label = f"{display_name} ({stage_label})"
            is_best = (stage_dir == "eval_reranked")

            marker = " ***" if is_best and o.get('hit@5', 0) > GAT_RESULTS['hit@5'] else ""
            print(f"{label:<35} {o.get('hit@1', 0):7.2f} {o.get('hit@5', 0):7.2f} "
                  f"{o.get('hit@10', 0):7.2f} {o.get('hit@20', 0):7.2f}{marker}")

            if is_best and (best_result is None or o.get('hit@5', 0) > best_result.get('hit@5', 0)):
                best_result = o

    print("-" * 85)

    if best_result:
        print(f"\nBest result vs GAT:")
        for k in ['hit@1', 'hit@5', 'hit@10', 'hit@20']:
            ours = best_result.get(k, 0)
            gat = GAT_RESULTS[k]
            delta = ours - gat
            print(f"  {k}: {ours:.2f} vs {gat:.2f} ({delta:+.2f})")


def per_repo_analysis():
    """Per-repo comparison between base and best pipeline."""
    base = load_summary("exp1_sft_only", "eval_filetree")
    best = load_summary("exp1_sft_only", "eval_reranked")

    if not base or not best:
        print("Cannot run per-repo analysis (missing summaries)")
        return

    print("\n" + "=" * 85)
    print("PER-REPO IMPROVEMENT (base -> full pipeline)")
    print("=" * 85)
    print(f"{'Repo':<25} {'N':>4} {'Base H@5':>9} {'Best H@5':>9} {'Delta':>7}")
    print("-" * 60)

    rows = []
    for repo in sorted(best['per_repo'].keys()):
        if repo not in base['per_repo']:
            continue
        b = base['per_repo'][repo]
        r = best['per_repo'][repo]
        n = r.get('count', 0)
        b_h5 = b.get('hit@5', 0)
        r_h5 = r.get('hit@5', 0)
        rows.append((repo, n, b_h5, r_h5, r_h5 - b_h5))

    rows.sort(key=lambda x: x[4], reverse=True)
    for repo, n, b_h5, r_h5, delta in rows:
        if n < 3:
            continue
        print(f"{repo:<25} {n:4d} {b_h5:9.1f} {r_h5:9.1f} {delta:+7.1f}")


def signal_contribution():
    """Load and display ablation results."""
    ablation_path = os.path.join(EXPERIMENTS_DIR, "exp1_sft_only", "ablation", "ablation_results.json")
    if not os.path.exists(ablation_path):
        print("Ablation results not found")
        return

    with open(ablation_path) as f:
        ablation = json.load(f)

    print("\n" + "=" * 85)
    print("ABLATION: Signal Contributions")
    print("=" * 85)
    print(f"{'Method':<35} {'H@1':>7} {'H@5':>7} {'H@10':>7} {'H@20':>7}")
    print("-" * 85)
    for r in ablation:
        print(f"{r['method']:<35} {r['hit@1']:7.2f} {r['hit@5']:7.2f} "
              f"{r['hit@10']:7.2f} {r['hit@20']:7.2f}")


def main():
    os.chdir("/home/chenlibin/grepo_agent")
    print_comparison_table()
    signal_contribution()
    per_repo_analysis()


if __name__ == "__main__":
    main()
