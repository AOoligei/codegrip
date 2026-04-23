#!/usr/bin/env python3
"""
Multi-pool neural fusion for SWE-bench Lite.

After running the neural reranker on multiple BM25 pools separately,
this script fuses the reranked lists using RRF to produce a single ranking.

The key insight: different BM25 pools surface different files.
Neural reranking on each pool gives complementary rankings.
RRF fusion combines them without needing score calibration.

Usage:
    python scripts/swebench_multipool_fusion.py
"""
import os
import json
from collections import defaultdict

import numpy as np


EXCLUDED_274 = {
    'astropy__astropy-7746', 'django__django-11039', 'django__django-12286',
    'django__django-12453', 'django__django-12983', 'django__django-13230',
    'django__django-13265', 'django__django-13321', 'django__django-13757',
    'django__django-13768', 'django__django-14016', 'django__django-14238',
    'django__django-14672', 'django__django-14730', 'django__django-14787',
    'django__django-15498', 'django__django-15789', 'django__django-16408',
    'matplotlib__matplotlib-23314', 'psf__requests-3362', 'pylint-dev__pylint-7114',
    'pytest-dev__pytest-5692', 'scikit-learn__scikit-learn-13241',
    'scikit-learn__scikit-learn-14894', 'sympy__sympy-12454', 'sympy__sympy-21614',
}


def load_predictions(pred_path):
    """Load predictions from a predictions.jsonl file."""
    preds = {}
    if not os.path.exists(pred_path):
        return preds
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            preds[key] = d.get('predicted', d.get('reranked', []))
    return preds


def rrf_fusion(rankings, weights=None, k=60):
    """Reciprocal Rank Fusion of multiple rankings."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] += w / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load all available reranked predictions
    rerank_sources = {
        'path_bm25': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_500/predictions.jsonl'),
        'content_bm25': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_content_bm25/predictions.jsonl'),
        'function_bm25': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_function_bm25/predictions.jsonl'),
        'funcnotest': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_funcnotest/predictions.jsonl'),
        'chunk': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_chunk/predictions.jsonl'),
        'best_ensemble': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_best_ensemble/predictions.jsonl'),
        'best_ens_1024': os.path.join(base_dir, 'experiments/rankft_runB_graph/eval_swebench_best_ensemble_1024/predictions.jsonl'),
        'runA_best_ens': os.path.join(base_dir, 'experiments/rankft_runA_bm25only/eval_swebench_best_ensemble/predictions.jsonl'),
    }

    # Also load BM25-only rankings for hybrid fusion
    bm25_sources = {
        'bm25_tricked': os.path.join(base_dir, 'data/rankft/swebench_bm25_final_top500.jsonl'),
        'bm25_funcnotest': os.path.join(base_dir, 'data/rankft/swebench_bm25_function_notest_top500.jsonl'),
        'bm25_chunk': os.path.join(base_dir, 'data/rankft/swebench_bm25_chunk_top500.jsonl'),
    }

    # Load GT
    gt_map = {}
    test_path = os.path.join(base_dir, 'data/swebench_lite/swebench_lite_test.jsonl')
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            gt_map[key] = set(d.get('changed_py_files', []))

    # Load neural reranked predictions
    neural_preds = {}
    available = []
    for name, path in rerank_sources.items():
        preds = load_predictions(path)
        if preds:
            neural_preds[name] = preds
            available.append(name)
            print(f"  Loaded {name}: {len(preds)} predictions")
    print(f"\n  Available neural sources: {available}")

    # Load BM25 rankings
    bm25_preds = {}
    for name, path in bm25_sources.items():
        if os.path.exists(path):
            preds = {}
            for l in open(path):
                d = json.loads(l)
                key = d.get('issue_id', d.get('instance_id', ''))
                preds[key] = d.get('bm25_candidates', [])
            bm25_preds[name] = preds
            print(f"  Loaded BM25 {name}: {len(preds)} rankings")

    if not available:
        print("\nNo neural reranking results available yet. Exiting.")
        return

    # Get common keys
    all_keys = set(gt_map.keys())
    for name in available:
        all_keys &= set(neural_preds[name].keys())
    all_keys = sorted(all_keys)
    print(f"\n  Common instances: {len(all_keys)}")

    # ============================================================
    # Strategy 1: RRF of all neural sources
    # ============================================================
    if len(available) >= 2:
        print("\n=== Multi-Pool Neural RRF ===")
        for rrf_k in [30, 60]:
            accs = {k: 0 for k in [1, 3, 5, 10, 20]}
            for key in all_keys:
                gt = gt_map.get(key, set())
                if not gt:
                    continue
                rankings = [neural_preds[name][key] for name in available]
                fused = rrf_fusion(rankings, k=rrf_k)
                for k_val in accs:
                    if gt.issubset(set(fused[:k_val])):
                        accs[k_val] += 1
            n = len(all_keys)
            print(f"  RRF(k={rrf_k}): " + " ".join(f"@{k}={v/n*100:.2f}%" for k, v in sorted(accs.items())))

            # Matched 274
            matched_keys = [k for k in all_keys if k not in EXCLUDED_274]
            accs_m = {k: 0 for k in [1, 5, 10]}
            for key in matched_keys:
                gt = gt_map.get(key, set())
                if not gt:
                    continue
                rankings = [neural_preds[name][key] for name in available]
                fused = rrf_fusion(rankings, k=rrf_k)
                for k_val in accs_m:
                    if gt.issubset(set(fused[:k_val])):
                        accs_m[k_val] += 1
            n_m = len(matched_keys)
            print(f"    274: " + " ".join(f"@{k}={v/n_m*100:.2f}%" for k, v in sorted(accs_m.items())))

    # ============================================================
    # Strategy 2: Hybrid (neural + BM25) fusion
    # ============================================================
    print("\n=== Neural + BM25 Hybrid ===")
    for neural_name in available:
        for bm25_name, bm25_pool in bm25_preds.items():
            accs = {k: 0 for k in [1, 5, 10, 20]}
            count = 0
            for key in all_keys:
                gt = gt_map.get(key, set())
                if not gt or key not in bm25_pool:
                    continue
                neural_ranking = neural_preds[neural_name][key]
                bm25_ranking = bm25_pool[key]
                fused = rrf_fusion([neural_ranking, bm25_ranking], weights=[2.0, 1.0], k=60)
                count += 1
                for k_val in accs:
                    if gt.issubset(set(fused[:k_val])):
                        accs[k_val] += 1
            if count > 0:
                print(f"  {neural_name} + {bm25_name}: " +
                      " ".join(f"@{k}={v/count*100:.2f}%" for k, v in sorted(accs.items())))

    # ============================================================
    # Strategy 3: All sources fusion (neural + BM25)
    # ============================================================
    if len(available) >= 2:
        print("\n=== All Sources Fusion ===")
        for rrf_k in [30, 60]:
            accs = {k: 0 for k in [1, 3, 5, 10, 20]}
            for key in all_keys:
                gt = gt_map.get(key, set())
                if not gt:
                    continue
                all_rankings = []
                weights = []
                for name in available:
                    all_rankings.append(neural_preds[name][key])
                    weights.append(2.0)
                for bm25_name, bm25_pool in bm25_preds.items():
                    if key in bm25_pool:
                        all_rankings.append(bm25_pool[key])
                        weights.append(1.0)
                fused = rrf_fusion(all_rankings, weights=weights, k=rrf_k)
                for k_val in accs:
                    if gt.issubset(set(fused[:k_val])):
                        accs[k_val] += 1
            n = len(all_keys)
            print(f"  All sources RRF(k={rrf_k}): " + " ".join(f"@{k}={v/n*100:.2f}%" for k, v in sorted(accs.items())))

    print("\nDone.")


if __name__ == '__main__':
    main()
