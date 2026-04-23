"""
Oracle / Upper-Bound Analysis for bug localization predictions.

Computes:
1. Oracle ceiling: what's the best Hit@K achievable with perfect reranking
   of the expanded candidate pool?
2. Pool coverage: what fraction of GT files are in the expansion pool?
3. Bottleneck decomposition: where exactly do we lose GT files?
4. Per-repo oracle analysis: which repos benefit most from better ranking?
5. Signal attribution: which expansion signal would have recovered each GT file?

Usage:
    python src/eval/oracle_analysis.py \
        --base_predictions experiments/exp1_sft_only/eval_filetree/predictions.jsonl \
        --expanded_predictions experiments/exp1_sft_only/eval_unified_expansion/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir experiments/exp1_sft_only/oracle_analysis
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_predictions(path: str) -> Dict[Tuple[str, int], dict]:
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            preds[key] = p
    return preds


def oracle_rerank(predicted: List[str], gt_set: Set[str]) -> List[str]:
    """Move GT files to front, preserve order otherwise."""
    gt_in_pred = [f for f in predicted if f in gt_set]
    non_gt = [f for f in predicted if f not in gt_set]
    return gt_in_pred + non_gt


def compute_hit_at_k(predicted: List[str], gt_set: Set[str], k: int) -> float:
    if not gt_set:
        return 0.0
    topk = set(predicted[:k])
    hits = len(gt_set & topk)
    return (hits / len(gt_set)) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_predictions', required=True,
                        help='Path to base (pre-expansion) predictions')
    parser.add_argument('--expanded_predictions', required=True,
                        help='Path to expanded predictions')
    parser.add_argument('--reranked_predictions', default=None,
                        help='Path to reranked predictions (optional)')
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    base_preds = load_predictions(args.base_predictions)
    exp_preds = load_predictions(args.expanded_predictions)
    reranked_preds = load_predictions(args.reranked_predictions) if args.reranked_predictions else None

    test_data = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_data[key] = item

    print(f"Loaded {len(base_preds)} base, {len(exp_preds)} expanded predictions")
    print(f"Test set: {len(test_data)} examples\n")

    # === 1. Bottleneck Decomposition ===
    print("=" * 60)
    print("1. BOTTLENECK DECOMPOSITION")
    print("=" * 60)

    categories = {
        'in_top5': 0,
        'in_top10_not_top5': 0,
        'in_pool_bad_rank': 0,  # in expanded but rank > 10
        'not_in_pool': 0,       # not in expanded predictions at all
    }
    total_gt = 0
    per_repo_decomp = defaultdict(lambda: defaultdict(int))

    for key, test_item in test_data.items():
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue

        expanded = exp_preds.get(key, {}).get('predicted', [])
        expanded_set = set(expanded)

        for f in gt:
            total_gt += 1
            repo = key[0]
            if f in set(expanded[:5]):
                categories['in_top5'] += 1
                per_repo_decomp[repo]['in_top5'] += 1
            elif f in set(expanded[:10]):
                categories['in_top10_not_top5'] += 1
                per_repo_decomp[repo]['in_top10_not_top5'] += 1
            elif f in expanded_set:
                categories['in_pool_bad_rank'] += 1
                per_repo_decomp[repo]['in_pool_bad_rank'] += 1
            else:
                categories['not_in_pool'] += 1
                per_repo_decomp[repo]['not_in_pool'] += 1

    print(f"\nTotal GT files: {total_gt}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_gt
        print(f"  {cat:25s}: {count:5d} ({pct:5.1f}%)")

    # === 2. Oracle Ceiling ===
    print(f"\n{'=' * 60}")
    print("2. ORACLE CEILING (perfect reranking of expansion pool)")
    print("=" * 60)

    ks = [1, 3, 5, 10, 20]

    # Compute actual vs oracle for each stage
    stages = [
        ("Base (pre-expansion)", base_preds),
        ("Expanded", exp_preds),
    ]
    if reranked_preds:
        stages.append(("Reranked", reranked_preds))

    results = {}
    for stage_name, preds in stages:
        actual = {k: [] for k in ks}
        oracle = {k: [] for k in ks}

        for key, test_item in test_data.items():
            gt = set(test_item.get('changed_py_files', []))
            if not gt:
                continue
            pred = preds.get(key, {}).get('predicted', [])
            oracle_pred = oracle_rerank(pred, gt)

            for k in ks:
                actual[k].append(compute_hit_at_k(pred, gt, k))
                oracle[k].append(compute_hit_at_k(oracle_pred, gt, k))

        actual_means = {k: sum(v) / len(v) for k, v in actual.items()}
        oracle_means = {k: sum(v) / len(v) for k, v in oracle.items()}
        results[stage_name] = {'actual': actual_means, 'oracle': oracle_means}

    # Print comparison table
    header = f"{'Stage':30s} | {'Type':8s}"
    for k in ks:
        header += f" | {'H@' + str(k):>7s}"
    print(header)
    print("-" * len(header))

    for stage_name in [s[0] for s in stages]:
        r = results[stage_name]
        line_actual = f"{stage_name:30s} | {'Actual':8s}"
        line_oracle = f"{'':30s} | {'Oracle':8s}"
        line_gap = f"{'':30s} | {'Gap':8s}"
        for k in ks:
            line_actual += f" | {r['actual'][k]:7.2f}"
            line_oracle += f" | {r['oracle'][k]:7.2f}"
            line_gap += f" | {r['oracle'][k] - r['actual'][k]:+7.2f}"
        print(line_actual)
        print(line_oracle)
        print(line_gap)
        print()

    # === 3. Per-Repo Oracle Analysis ===
    print(f"{'=' * 60}")
    print("3. PER-REPO ANALYSIS (top 15 repos by oracle gap)")
    print("=" * 60)

    repo_oracle_gaps = []
    repo_stats = defaultdict(lambda: {'actual_h5': [], 'oracle_h5': [], 'count': 0})

    for key, test_item in test_data.items():
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue
        repo = key[0]
        pred = exp_preds.get(key, {}).get('predicted', [])
        oracle_pred = oracle_rerank(pred, gt)

        repo_stats[repo]['actual_h5'].append(compute_hit_at_k(pred, gt, 5))
        repo_stats[repo]['oracle_h5'].append(compute_hit_at_k(oracle_pred, gt, 5))
        repo_stats[repo]['count'] += 1

    for repo, stats in repo_stats.items():
        actual = sum(stats['actual_h5']) / len(stats['actual_h5'])
        oracle = sum(stats['oracle_h5']) / len(stats['oracle_h5'])
        gap = oracle - actual
        repo_oracle_gaps.append((repo, actual, oracle, gap, stats['count']))

    repo_oracle_gaps.sort(key=lambda x: -x[3])
    print(f"\n{'Repo':40s} | {'Actual H@5':>10s} | {'Oracle H@5':>10s} | {'Gap':>7s} | {'N':>4s}")
    print("-" * 80)
    for repo, actual, oracle, gap, count in repo_oracle_gaps[:15]:
        print(f"{repo:40s} | {actual:10.2f} | {oracle:10.2f} | {gap:+7.2f} | {count:4d}")

    # === 4. Signal Attribution ===
    print(f"\n{'=' * 60}")
    print("4. SIGNAL ATTRIBUTION (how expansion recovers GT files)")
    print("=" * 60)

    base_found = 0
    expansion_recovered = 0
    never_found = 0

    for key, test_item in test_data.items():
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue
        base = set(base_preds.get(key, {}).get('predicted', []))
        expanded = set(exp_preds.get(key, {}).get('predicted', []))

        for f in gt:
            if f in base:
                base_found += 1
            elif f in expanded:
                expansion_recovered += 1
            else:
                never_found += 1

    print(f"\n  Found by base model:      {base_found:5d} ({100 * base_found / total_gt:.1f}%)")
    print(f"  Recovered by expansion:   {expansion_recovered:5d} ({100 * expansion_recovered / total_gt:.1f}%)")
    print(f"  Never found:              {never_found:5d} ({100 * never_found / total_gt:.1f}%)")

    # === 5. Save detailed results ===
    output = {
        'bottleneck_decomposition': categories,
        'total_gt_files': total_gt,
        'oracle_results': results,
        'per_repo_oracle': {
            repo: {'actual_h5': actual, 'oracle_h5': oracle, 'gap': gap, 'count': count}
            for repo, actual, oracle, gap, count in repo_oracle_gaps
        },
        'signal_attribution': {
            'base_found': base_found,
            'expansion_recovered': expansion_recovered,
            'never_found': never_found,
        }
    }
    output_path = os.path.join(args.output_dir, 'oracle_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
