"""
Ensemble predictions from multiple experiments.
Combines predictions by rank-based fusion or vote-based merging.

Usage:
    python src/eval/ensemble_predictions.py \
        --predictions exp1/predictions.jsonl exp5/predictions.jsonl \
        --output ensemble/predictions.jsonl \
        --method rank_fusion
"""

import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def load_predictions(path: str) -> Dict[Tuple[str, int], dict]:
    """Load predictions indexed by (repo, issue_id)."""
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            preds[key] = p
    return preds


def reciprocal_rank_fusion(
    pred_lists: List[Dict[Tuple[str, int], dict]],
    weights: List[float] = None,
    k: int = 60,
) -> Dict[Tuple[str, int], List[str]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple prediction lists.
    Each file gets score = sum(weight_i / (rank_i + k)) across models.
    """
    if weights is None:
        weights = [1.0] * len(pred_lists)

    # Get all issue keys
    all_keys = set()
    for preds in pred_lists:
        all_keys.update(preds.keys())

    fused = {}
    for key in all_keys:
        file_scores: Dict[str, float] = defaultdict(float)

        for preds, weight in zip(pred_lists, weights):
            if key not in preds:
                continue
            predicted = preds[key]['predicted']
            for rank, f in enumerate(predicted):
                file_scores[f] += weight / (rank + k)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        fused[key] = [f for f, _ in sorted_files]

    return fused


def vote_fusion(
    pred_lists: List[Dict[Tuple[str, int], dict]],
    weights: List[float] = None,
    top_k: int = 10,
) -> Dict[Tuple[str, int], List[str]]:
    """
    Vote-based fusion: files predicted by more models get higher rank.
    Within same vote count, rank by average position.
    """
    if weights is None:
        weights = [1.0] * len(pred_lists)

    all_keys = set()
    for preds in pred_lists:
        all_keys.update(preds.keys())

    fused = {}
    for key in all_keys:
        file_votes: Dict[str, float] = defaultdict(float)
        file_positions: Dict[str, List[float]] = defaultdict(list)

        for preds, weight in zip(pred_lists, weights):
            if key not in preds:
                continue
            predicted = preds[key]['predicted'][:top_k]
            for rank, f in enumerate(predicted):
                file_votes[f] += weight
                file_positions[f].append(rank)

        # Sort by votes (desc), then by avg position (asc)
        items = list(file_votes.keys())
        items.sort(key=lambda f: (-file_votes[f], sum(file_positions[f]) / len(file_positions[f])))
        fused[key] = items

    return fused


def interleave_fusion(
    pred_lists: List[Dict[Tuple[str, int], dict]],
) -> Dict[Tuple[str, int], List[str]]:
    """
    Round-robin interleaving: take files alternately from each model.
    Preserves the ordering from each model while diversifying.
    """
    all_keys = set()
    for preds in pred_lists:
        all_keys.update(preds.keys())

    fused = {}
    for key in all_keys:
        seen = set()
        result = []
        lists = [preds[key]['predicted'] if key in preds else [] for preds in pred_lists]
        max_len = max(len(l) for l in lists) if lists else 0

        for i in range(max_len):
            for l in lists:
                if i < len(l) and l[i] not in seen:
                    result.append(l[i])
                    seen.add(l[i])

        fused[key] = result

    return fused


def evaluate_fusion(
    fused: Dict[Tuple[str, int], List[str]],
    reference: Dict[Tuple[str, int], dict],
    output_path: str,
) -> dict:
    """Evaluate fused predictions and write results."""
    expanded_preds = []

    for key in sorted(fused.keys()):
        ref = reference.get(key)
        if ref is None:
            continue

        predicted = fused[key]
        gt = set(ref.get('changed_py_files', ref.get('ground_truth', [])))

        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            topk = set(predicted[:k])
            hits = len(gt & topk)
            metrics[f'hit@{k}'] = (hits / len(gt)) * 100 if gt else 0.0

        expanded_preds.append({
            'repo': key[0],
            'issue_id': key[1],
            'ground_truth': list(gt),
            'predicted': predicted,
            'metrics': metrics,
        })

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for p in expanded_preds:
            f.write(json.dumps(p) + '\n')

    # Summary
    repo_metrics = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    for p in expanded_preds:
        for k, v in p['metrics'].items():
            repo_metrics[p['repo']][k].append(v)
            overall[k].append(v)

    summary = {
        'overall': {k: sum(v) / len(v) for k, v in overall.items()},
        'per_repo': {}
    }
    for repo, metrics in repo_metrics.items():
        summary['per_repo'][repo] = {k: sum(v) / len(v) for k, v in metrics.items()}
        summary['per_repo'][repo]['count'] = len(metrics.get('hit@1', []))

    summary_path = output_path.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    o = summary['overall']
    print(f"Hit@1={o['hit@1']:.2f}% Hit@3={o['hit@3']:.2f}% Hit@5={o['hit@5']:.2f}% "
          f"Hit@10={o['hit@10']:.2f}% Hit@20={o['hit@20']:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--weights', nargs='+', type=float, default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['rrf', 'vote', 'interleave'], default='rrf')
    parser.add_argument('--rrf_k', type=int, default=60)
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    args = parser.parse_args()

    # Load all prediction sets
    pred_lists = []
    for path in args.predictions:
        pred_lists.append(load_predictions(path))
        print(f"Loaded {len(pred_lists[-1])} predictions from {path}")

    # Load reference data for evaluation
    reference = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            reference[key] = item

    # Fuse
    if args.method == 'rrf':
        print(f"Fusing with Reciprocal Rank Fusion (k={args.rrf_k})...")
        fused = reciprocal_rank_fusion(pred_lists, args.weights, k=args.rrf_k)
    elif args.method == 'vote':
        print("Fusing with vote-based method...")
        fused = vote_fusion(pred_lists, args.weights)
    elif args.method == 'interleave':
        print("Fusing with interleaving...")
        fused = interleave_fusion(pred_lists)

    # Evaluate
    print(f"\nEnsemble results ({len(fused)} predictions):")
    evaluate_fusion(fused, reference, args.output)


if __name__ == '__main__':
    main()
