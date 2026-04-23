"""
Co-change expansion: enhance SFT predictions by adding files that frequently
co-change with predicted files (from training PR data).

This should improve recall (Hit@5/10/20) without hurting precision (Hit@1).
"""

import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def build_cochange_index(train_data_path: str, min_cochange: int = 2) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build per-repo co-change index from training PR data.
    Returns: {repo: {file_a: {file_b: score}}}
    where score = co-change_count(a,b) / total_changes(a)
    """
    # Count per-repo co-changes
    repo_cochanges: Dict[str, Counter] = defaultdict(Counter)
    repo_file_count: Dict[str, Counter] = defaultdict(Counter)

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get('split') != 'train':
                continue
            repo = item['repo']
            files = item.get('changed_py_files', [])
            if not files:
                files = [f for f in item.get('changed_files', []) if f.endswith('.py')]

            for f in files:
                repo_file_count[repo][f] += 1

            # Count co-changes between all pairs
            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    # Build normalized index
    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                # Normalized score: how often fb changes when fa changes
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score

    return index


def expand_predictions(
    predictions_path: str,
    cochange_index: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    max_expand: int = 15,
    min_score: float = 0.1,
) -> None:
    """
    Expand SFT predictions using co-change patterns.
    Keeps original predictions first, then appends co-change expansions.
    """
    preds = []
    with open(predictions_path) as f:
        for line in f:
            preds.append(json.loads(line))

    expanded_preds = []
    expand_count = 0

    for p in preds:
        repo = p['repo']
        original = list(p['predicted'])
        original_set = set(original)

        # Collect co-change candidates
        candidates: Dict[str, float] = {}
        repo_idx = cochange_index.get(repo, {})

        for pred_file in original:
            neighbors = repo_idx.get(pred_file, {})
            for neighbor, score in neighbors.items():
                if neighbor not in original_set and score >= min_score:
                    candidates[neighbor] = max(candidates.get(neighbor, 0), score)

        # Sort candidates by score and add top ones
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        expansion = [c[0] for c in sorted_cands[:max_expand]]

        if expansion:
            expand_count += 1

        new_predicted = original + expansion

        # Recompute metrics
        gt = set(p['ground_truth'])
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            topk = set(new_predicted[:k])
            hits = len(gt & topk)
            metrics[f'hit@{k}'] = (hits / len(gt)) * 100 if gt else 0.0

        new_p = dict(p)
        new_p['predicted'] = new_predicted
        new_p['predicted_original'] = original
        new_p['metrics'] = metrics
        new_p['num_expanded'] = len(expansion)
        expanded_preds.append(new_p)

    # Write expanded predictions
    with open(output_path, 'w') as f:
        for p in expanded_preds:
            f.write(json.dumps(p) + '\n')

    # Compute and save summary
    summary = compute_summary(expanded_preds)
    summary_path = output_path.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results
    o = summary['overall']
    print(f"Expanded {expand_count}/{len(preds)} predictions")
    print(f"Hit@1={o['hit@1']:.2f}% Hit@3={o['hit@3']:.2f}% Hit@5={o['hit@5']:.2f}% "
          f"Hit@10={o['hit@10']:.2f}% Hit@20={o['hit@20']:.2f}%")


def compute_summary(preds: List[dict]) -> dict:
    """Compute per-repo and overall metrics."""
    from collections import defaultdict

    repo_metrics = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for p in preds:
        repo = p['repo']
        for k, v in p['metrics'].items():
            repo_metrics[repo][k].append(v)
            overall[k].append(v)
        repo_metrics[repo]['count'] = [len(repo_metrics[repo].get('hit@1', []))]

    summary = {
        'overall': {k: sum(v) / len(v) for k, v in overall.items()},
        'per_repo': {}
    }

    for repo, metrics in repo_metrics.items():
        summary['per_repo'][repo] = {}
        for k, v in metrics.items():
            if k == 'count':
                summary['per_repo'][repo][k] = len(repo_metrics[repo].get('hit@1', []))
            else:
                summary['per_repo'][repo][k] = sum(v) / len(v)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='Path to predictions.jsonl')
    parser.add_argument('--train_data', required=True, help='Path to train+test JSONL')
    parser.add_argument('--output', required=True, help='Output predictions path')
    parser.add_argument('--min_cochange', type=int, default=2)
    parser.add_argument('--min_score', type=float, default=0.1)
    parser.add_argument('--max_expand', type=int, default=15)
    args = parser.parse_args()

    print("Building co-change index...")
    # Load all data (train split only used for co-change)
    index = build_cochange_index(args.train_data, min_cochange=args.min_cochange)

    total_files = sum(len(v) for repo in index.values() for v in repo.values())
    print(f"Co-change index: {len(index)} repos, {total_files} file pairs")

    print("Expanding predictions...")
    expand_predictions(
        args.predictions, index, args.output,
        max_expand=args.max_expand,
        min_score=args.min_score,
    )


if __name__ == '__main__':
    main()
