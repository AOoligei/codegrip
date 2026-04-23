"""
Ensemble CodeGRIP predictions with BM25/TF-IDF retrieval scores.
Uses Reciprocal Rank Fusion (RRF) to combine model and retrieval rankings.

Usage:
    python src/eval/ensemble_with_retrieval.py \
        --model_predictions experiments/exp1_sft_only/eval_reranked/predictions.jsonl \
        --retrieval_predictions experiments/baselines/combined/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output experiments/exp1_sft_only/eval_ensemble_retrieval/predictions.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


def compute_hit_at_k(predicted: List[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def ensemble_rrf(model_ranked: List[str], retrieval_ranked: List[str],
                 w_model: float = 1.0, w_retrieval: float = 0.5,
                 max_candidates: int = 20) -> List[str]:
    """Combine two ranked lists using weighted RRF."""
    scores = defaultdict(float)

    for rank, f in enumerate(model_ranked, start=1):
        scores[f] += w_model * rrf_score(rank)

    for rank, f in enumerate(retrieval_ranked, start=1):
        scores[f] += w_retrieval * rrf_score(rank)

    ranked = sorted(scores.keys(), key=lambda f: -scores[f])
    return ranked[:max_candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_predictions', required=True,
                        help='CodeGRIP predictions (after expansion/reranking)')
    parser.add_argument('--retrieval_predictions', required=True,
                        help='BM25/retrieval baseline predictions')
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--w_model', type=float, default=1.0)
    parser.add_argument('--w_retrieval', type=float, default=0.5)
    args = parser.parse_args()

    # Load model predictions
    model_preds = {}
    with open(args.model_predictions) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            model_preds[key] = p

    # Load retrieval predictions
    retrieval_preds = {}
    with open(args.retrieval_predictions) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            retrieval_preds[key] = p

    # Load test data for ground truth
    test_data = {}
    with open(args.test_data) as f:
        for line in f:
            ex = json.loads(line)
            key = (ex['repo'], ex['issue_id'])
            test_data[key] = ex

    # Ensemble
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}

    for key, mpred in model_preds.items():
        gt = set(mpred.get('ground_truth', []))
        if not gt and key in test_data:
            gt = set(test_data[key].get('changed_py_files', []))

        model_ranked = mpred.get('predicted', [])
        retrieval_ranked = retrieval_preds.get(key, {}).get('predicted', [])

        # RRF ensemble
        ensembled = ensemble_rrf(
            model_ranked, retrieval_ranked,
            w_model=args.w_model, w_retrieval=args.w_retrieval
        )

        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ensembled, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)

        results.append({
            'repo': mpred['repo'],
            'issue_id': mpred['issue_id'],
            'ground_truth': list(gt),
            'predicted': ensembled,
            'metrics': metrics,
            'method': 'rrf_ensemble',
        })

    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # Summary
    summary = {
        'method': 'rrf_ensemble',
        'w_model': args.w_model,
        'w_retrieval': args.w_retrieval,
        'num_examples': len(results),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
    }

    summary_path = args.output.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Ensemble results ({len(results)} examples):")
    for k, v in summary['metrics'].items():
        print(f"  {k}: {v}")
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
