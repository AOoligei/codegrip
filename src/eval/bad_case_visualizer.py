"""
Automated Bad Case Visualizer for bug localization predictions.

Generates a structured report of failure cases with:
1. Worst-performing repos (by Hit@5)
2. Completely missed issues (0 GT files found)
3. Near-miss cases (GT files in pool but ranked poorly)
4. Structural analysis of failures
5. Markdown report output

Usage:
    python src/eval/bad_case_visualizer.py \
        --predictions experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --train_data data/grepo_text/grepo_train.jsonl \
        --output experiments/exp1_sft_only/bad_case_report.md
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def load_predictions(path: str) -> Dict[Tuple[str, int], dict]:
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            preds[key] = p
    return preds


def compute_hit_at_k(predicted: List[str], gt_set: Set[str], k: int) -> float:
    if not gt_set:
        return 0.0
    topk = set(predicted[:k])
    return (len(gt_set & topk) / len(gt_set)) * 100


def build_cochange_index(train_data: List[dict]) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo, per-file co-change neighbors."""
    index = defaultdict(lambda: defaultdict(set))
    for item in train_data:
        repo = item['repo']
        files = item.get('changed_py_files', [])
        for f in files:
            for other in files:
                if other != f:
                    index[repo][f].add(other)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of examples per category')
    args = parser.parse_args()

    # Load data
    preds = load_predictions(args.predictions)
    test_items = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_items[key] = item

    with open(args.train_data) as f:
        train_data = [json.loads(l) for l in f]
    cochange = build_cochange_index(train_data)

    # Compute per-example metrics
    examples = []
    for key, test_item in test_items.items():
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue
        pred = preds.get(key, {}).get('predicted', [])
        pred_set = set(pred)

        h1 = compute_hit_at_k(pred, gt, 1)
        h5 = compute_hit_at_k(pred, gt, 5)
        h10 = compute_hit_at_k(pred, gt, 10)

        missed = [f for f in gt if f not in pred_set]
        found = [f for f in gt if f in pred_set]
        found_positions = {f: pred.index(f) + 1 for f in found}

        # Check structural connectivity of missed files
        repo = key[0]
        missed_analysis = []
        for mf in missed:
            cc_neighbors = cochange.get(repo, {}).get(mf, set())
            connected_to_pred = cc_neighbors & pred_set
            same_dir_in_pred = [p for p in pred if os.path.dirname(p) == os.path.dirname(mf)]
            missed_analysis.append({
                'file': mf,
                'has_cochange_to_pred': len(connected_to_pred) > 0,
                'cochange_connections': list(connected_to_pred)[:3],
                'same_dir_count': len(same_dir_in_pred),
                'depth': mf.count('/'),
                'is_test': 'test' in os.path.basename(mf).lower(),
            })

        examples.append({
            'repo': key[0],
            'issue_id': key[1],
            'issue_text': test_item['issue_text'][:500],
            'gt_files': list(gt),
            'predicted_top10': pred[:10],
            'hit_at_1': h1,
            'hit_at_5': h5,
            'hit_at_10': h10,
            'n_gt': len(gt),
            'n_found': len(found),
            'n_missed': len(missed),
            'found_positions': found_positions,
            'missed_analysis': missed_analysis,
        })

    # === Generate Report ===
    lines = []
    lines.append("# Bad Case Analysis Report\n")
    lines.append(f"Total examples: {len(examples)}\n")

    # Overall stats
    total_h1 = sum(e['hit_at_1'] for e in examples) / len(examples)
    total_h5 = sum(e['hit_at_5'] for e in examples) / len(examples)
    total_h10 = sum(e['hit_at_10'] for e in examples) / len(examples)
    lines.append(f"Overall: H@1={total_h1:.2f}, H@5={total_h5:.2f}, H@10={total_h10:.2f}\n")

    # === Section 1: Worst repos ===
    lines.append("\n## 1. Worst-Performing Repos (by H@5)\n")
    repo_metrics = defaultdict(list)
    for e in examples:
        repo_metrics[e['repo']].append(e['hit_at_5'])

    repo_avg = [(repo, sum(v) / len(v), len(v)) for repo, v in repo_metrics.items()]
    repo_avg.sort(key=lambda x: x[1])

    lines.append("| Repo | Avg H@5 | N Examples |")
    lines.append("|------|---------|------------|")
    for repo, avg, n in repo_avg[:args.top_n]:
        lines.append(f"| {repo} | {avg:.1f} | {n} |")

    # === Section 2: Complete misses (H@10 = 0) ===
    complete_misses = [e for e in examples if e['hit_at_10'] == 0]
    lines.append(f"\n## 2. Complete Misses (H@10 = 0): {len(complete_misses)} examples\n")

    # Sample some
    for e in sorted(complete_misses, key=lambda x: -x['n_gt'])[:args.top_n]:
        lines.append(f"### {e['repo']} / Issue #{e['issue_id']}")
        lines.append(f"- GT files ({e['n_gt']}): {', '.join(e['gt_files'][:5])}")
        if e['n_gt'] > 5:
            lines.append(f"  ... and {e['n_gt'] - 5} more")
        lines.append(f"- Top-5 predicted: {', '.join(e['predicted_top10'][:5])}")
        lines.append(f"- Issue: {e['issue_text'][:200]}...")
        lines.append("")

        # Missed file analysis
        for ma in e['missed_analysis'][:3]:
            conn = "YES" if ma['has_cochange_to_pred'] else "NO"
            lines.append(f"  - `{ma['file']}`: co-change to pred={conn}, "
                        f"same-dir={ma['same_dir_count']}, depth={ma['depth']}, "
                        f"test={ma['is_test']}")
        lines.append("")

    # === Section 3: Near misses (found in pool but rank > 5) ===
    near_misses = [e for e in examples if e['hit_at_10'] > e['hit_at_5'] > 0]
    lines.append(f"\n## 3. Near Misses (in top-10 but not top-5): {len(near_misses)} examples\n")

    for e in sorted(near_misses, key=lambda x: x['n_gt'] - x['n_found'], reverse=True)[:args.top_n]:
        lines.append(f"### {e['repo']} / Issue #{e['issue_id']}")
        lines.append(f"- H@5={e['hit_at_5']:.0f}%, H@10={e['hit_at_10']:.0f}%")
        lines.append(f"- Found positions: {e['found_positions']}")
        lines.append(f"- Could improve by promoting rank 6-10 files to top-5")
        lines.append("")

    # === Section 4: Pattern analysis of missed files ===
    lines.append("\n## 4. Pattern Analysis of Missed Files\n")

    all_missed = []
    for e in examples:
        all_missed.extend(e['missed_analysis'])

    if all_missed:
        n_test = sum(1 for m in all_missed if m['is_test'])
        n_deep = sum(1 for m in all_missed if m['depth'] >= 4)
        n_cc = sum(1 for m in all_missed if m['has_cochange_to_pred'])
        n_same_dir = sum(1 for m in all_missed if m['same_dir_count'] > 0)
        total_missed = len(all_missed)

        lines.append(f"Total missed GT files: {total_missed}\n")
        lines.append("| Pattern | Count | % |")
        lines.append("|---------|-------|---|")
        lines.append(f"| Test files | {n_test} | {100 * n_test / total_missed:.1f}% |")
        lines.append(f"| Deep files (depth >= 4) | {n_deep} | {100 * n_deep / total_missed:.1f}% |")
        lines.append(f"| Has co-change to predictions | {n_cc} | {100 * n_cc / total_missed:.1f}% |")
        lines.append(f"| Same directory as predictions | {n_same_dir} | {100 * n_same_dir / total_missed:.1f}% |")
        lines.append(f"| No structural connection | {total_missed - n_cc - n_same_dir + sum(1 for m in all_missed if m['has_cochange_to_pred'] and m['same_dir_count'] > 0)} | - |")

    # === Section 5: Success patterns ===
    lines.append("\n## 5. Success Patterns (what we get right)\n")

    successes = [e for e in examples if e['hit_at_5'] == 100]
    lines.append(f"Perfect H@5 examples: {len(successes)}\n")

    if successes:
        # Analyze what makes successful cases easier
        avg_gt_success = sum(e['n_gt'] for e in successes) / len(successes)
        avg_gt_all = sum(e['n_gt'] for e in examples) / len(examples)
        lines.append(f"- Avg GT files (success): {avg_gt_success:.1f}")
        lines.append(f"- Avg GT files (all): {avg_gt_all:.1f}")

        single_file_success = sum(1 for e in successes if e['n_gt'] == 1)
        single_file_all = sum(1 for e in examples if e['n_gt'] == 1)
        lines.append(f"- Single-file issues (success): {single_file_success}/{len(successes)} "
                     f"({100 * single_file_success / len(successes):.0f}%)")
        lines.append(f"- Single-file issues (all): {single_file_all}/{len(examples)} "
                     f"({100 * single_file_all / len(examples):.0f}%)")

    # Write report
    report = "\n".join(lines)
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"Report saved to {args.output}")
    print(f"Total examples: {len(examples)}")
    print(f"Complete misses: {len(complete_misses)}")
    print(f"Near misses: {len(near_misses)}")
    print(f"Perfect H@5: {len(successes)}")


if __name__ == '__main__':
    main()
