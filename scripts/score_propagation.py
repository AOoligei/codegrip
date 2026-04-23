"""
Graph Score Propagation (Stage 4): Post-reranking score adjustment using graph structure.

After the cross-encoder scores candidates independently (pointwise), propagate
scores through the co-change/import graph among candidates. Intuition: if file A
scores high and its co-change neighbor B scores moderately, B should get boosted.

This implements label-propagation-style score adjustment:
  s^{(t+1)} = alpha * s_original + (1-alpha) * W_norm @ s^{(t)}

where W_norm is the row-normalized adjacency matrix among candidates.

Usage:
  python scripts/score_propagation.py \
    --predictions experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl \
    --train_data data/grepo_text/grepo_train.jsonl \
    --dep_graph_dir data/dep_graphs
"""

import json
import argparse
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Set


def build_cochange_index(train_data_path: str, min_cochange: int = 1) -> Dict:
    """Build per-repo co-change index from training PR data."""
    repo_cochanges = defaultdict(lambda: defaultdict(int))
    repo_file_count = defaultdict(lambda: defaultdict(int))

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get('split') != 'train':
                continue
            repo = item['repo']
            files = item.get('changed_py_files', [])
            if not files:
                files = [ff for ff in item.get('changed_files', []) if ff.endswith('.py')]

            for f_item in files:
                repo_file_count[repo][f_item] += 1

            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    index = {}
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score

    return index


def build_import_index(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo bidirectional import index from dep_graphs."""
    index = {}
    if not os.path.isdir(dep_graph_dir):
        return index

    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith('_rels.json'):
            continue
        repo = fname.replace('_rels.json', '')

        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)

        neighbors = defaultdict(set)
        for src, targets in rels.get('file_imports', {}).items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

        for src_func, callees in rels.get('call_graph', {}).items():
            src_file = src_func.split(':')[0] if ':' in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(':')[0] if ':' in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)

        index[repo] = dict(neighbors)

    return index


def build_adjacency_matrix(files, repo, cc_index, imp_index, cc_weight=1.0, imp_weight=0.5):
    """Build adjacency matrix among candidate files using graph edges."""
    n = len(files)
    adj = np.zeros((n, n))
    file_to_idx = {f: i for i, f in enumerate(files)}

    repo_cc = cc_index.get(repo, {})
    repo_imp = imp_index.get(repo, {})

    edges_found = 0
    for i, fi in enumerate(files):
        # Co-change edges
        for neighbor, score in repo_cc.get(fi, {}).items():
            if neighbor in file_to_idx:
                j = file_to_idx[neighbor]
                if i != j:
                    adj[i, j] = max(adj[i, j], cc_weight * score)
                    edges_found += 1

        # Import edges
        for neighbor in repo_imp.get(fi, set()):
            if neighbor in file_to_idx:
                j = file_to_idx[neighbor]
                if i != j:
                    adj[i, j] = max(adj[i, j], imp_weight)
                    edges_found += 1

    return adj, edges_found


def propagate_scores(scores, adj, alpha=0.5, iterations=5):
    """
    Label-propagation-style score adjustment.
    s^{(t+1)} = alpha * s_original + (1-alpha) * W_norm @ s^{(t)}

    alpha=1.0 means no propagation (keep original scores).
    alpha=0.0 means full graph smoothing.
    """
    s = np.array(scores, dtype=np.float64)
    s_original = s.copy()

    # Row-normalize adjacency
    row_sum = adj.sum(axis=1)
    # For nodes with no neighbors, keep their original score
    has_neighbors = row_sum > 0
    W_norm = np.zeros_like(adj)
    W_norm[has_neighbors] = adj[has_neighbors] / row_sum[has_neighbors, np.newaxis]

    for _ in range(iterations):
        propagated = W_norm @ s
        s = alpha * s_original + (1 - alpha) * propagated
        # For nodes with no neighbors, always keep original
        s[~has_neighbors] = s_original[~has_neighbors]

    return s


def compute_recall_at_k(predicted, ground_truth, k):
    """Compute recall@k: fraction of GT files in top-k predictions."""
    topk = set(predicted[:k])
    found = len(topk & set(ground_truth))
    return found / len(ground_truth) if ground_truth else 0.0


def evaluate_propagation(predictions, cc_index, imp_index, alpha, iterations,
                         cc_weight=1.0, imp_weight=0.5):
    """Run score propagation and compute metrics."""
    results = {
        'recall@1': [], 'recall@5': [], 'recall@10': [], 'recall@20': [],
        'edges_per_example': [], 'rank_changes': [],
        'improved': 0, 'degraded': 0, 'unchanged': 0,
    }

    for pred in predictions:
        files = pred['predicted']
        scores = pred['scores']
        repo = pred['repo']
        gt = pred['ground_truth']

        # Original recall@1
        orig_r1 = compute_recall_at_k(files, gt, 1)

        # Build adjacency matrix
        adj, n_edges = build_adjacency_matrix(files, repo, cc_index, imp_index,
                                               cc_weight, imp_weight)
        results['edges_per_example'].append(n_edges)

        if n_edges == 0 or alpha >= 1.0:
            # No graph edges or no propagation — keep original ranking
            new_files = files
        else:
            # Propagate scores
            new_scores = propagate_scores(scores, adj, alpha=alpha, iterations=iterations)

            # Re-rank by new scores
            sorted_indices = np.argsort(-new_scores)
            new_files = [files[i] for i in sorted_indices]

        # Compute new metrics
        new_r1 = compute_recall_at_k(new_files, gt, 1)

        for k in [1, 5, 10, 20]:
            results[f'recall@{k}'].append(compute_recall_at_k(new_files, gt, k))

        # Track changes
        rank_changed = sum(1 for i, f in enumerate(new_files) if i < len(files) and f != files[i])
        results['rank_changes'].append(rank_changed)

        if new_r1 > orig_r1:
            results['improved'] += 1
        elif new_r1 < orig_r1:
            results['degraded'] += 1
        else:
            results['unchanged'] += 1

    # Aggregate
    n = len(predictions)
    summary = {}
    for k in [1, 5, 10, 20]:
        summary[f'R@{k}'] = sum(results[f'recall@{k}']) / n * 100

    summary['avg_edges'] = sum(results['edges_per_example']) / n
    summary['avg_rank_changes'] = sum(results['rank_changes']) / n
    summary['improved'] = results['improved']
    summary['degraded'] = results['degraded']
    summary['unchanged'] = results['unchanged']

    return summary


def main():
    parser = argparse.ArgumentParser(description='Graph Score Propagation (Stage 4)')
    parser.add_argument('--predictions', required=True,
                        help='Path to predictions.jsonl from eval_merged_rerank')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl',
                        help='Training data for co-change graph')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs',
                        help='Directory with import graph files')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Retention weight (1.0=no change). If None, sweeps.')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of propagation iterations')
    parser.add_argument('--cc_weight', type=float, default=1.0,
                        help='Weight for co-change edges')
    parser.add_argument('--imp_weight', type=float, default=0.5,
                        help='Weight for import edges')
    args = parser.parse_args()

    # Load predictions
    print("Loading predictions...")
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            predictions.append(json.loads(line))
    print(f"  {len(predictions)} examples")

    # Build graph indices
    print("Building co-change index...")
    cc_index = build_cochange_index(args.train_data)
    print(f"  {len(cc_index)} repos")

    print("Building import index...")
    imp_index = build_import_index(args.dep_graph_dir)
    print(f"  {len(imp_index)} repos")

    # Baseline (no propagation)
    baseline = evaluate_propagation(predictions, cc_index, imp_index,
                                     alpha=1.0, iterations=0,
                                     cc_weight=args.cc_weight, imp_weight=args.imp_weight)
    print(f"\nBaseline (no propagation):")
    print(f"  R@1={baseline['R@1']:.2f}%  R@5={baseline['R@5']:.2f}%  "
          f"R@10={baseline['R@10']:.2f}%  R@20={baseline['R@20']:.2f}%")

    if args.alpha is not None:
        # Single alpha evaluation
        result = evaluate_propagation(predictions, cc_index, imp_index,
                                       alpha=args.alpha, iterations=args.iterations,
                                       cc_weight=args.cc_weight, imp_weight=args.imp_weight)
        print(f"\nalpha={args.alpha}, iterations={args.iterations}:")
        print(f"  R@1={result['R@1']:.2f}%  R@5={result['R@5']:.2f}%  "
              f"R@10={result['R@10']:.2f}%  R@20={result['R@20']:.2f}%")
        print(f"  Avg edges: {result['avg_edges']:.1f}, Avg rank changes: {result['avg_rank_changes']:.1f}")
        print(f"  Improved: {result['improved']}, Degraded: {result['degraded']}, "
              f"Unchanged: {result['unchanged']}")
    else:
        # Sweep alpha
        print(f"\n{'alpha':>8} {'iter':>4} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} "
              f"{'edges':>6} {'chg':>5} {'up':>4} {'dn':>4}")
        print("-" * 75)

        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            for iters in [1, 3, 5, 10]:
                result = evaluate_propagation(predictions, cc_index, imp_index,
                                               alpha=alpha, iterations=iters,
                                               cc_weight=args.cc_weight,
                                               imp_weight=args.imp_weight)
                delta = result['R@1'] - baseline['R@1']
                marker = "+" if delta > 0 else " "
                print(f"{alpha:>8.2f} {iters:>4d} {result['R@1']:>7.2f}% {result['R@5']:>7.2f}% "
                      f"{result['R@10']:>7.2f}% {result['R@20']:>7.2f}% "
                      f"{result['avg_edges']:>5.1f} {result['avg_rank_changes']:>5.1f} "
                      f"{result['improved']:>4d} {result['degraded']:>4d} {marker}{delta:+.2f}")

        # Also try edge-type-specific propagation
        print(f"\n--- Edge-type specific ---")
        print(f"{'config':>20} {'R@1':>8} {'R@5':>8} {'delta':>8}")
        print("-" * 50)

        for cc_w, imp_w, label in [(1.0, 0.0, "co-change only"),
                                    (0.0, 0.5, "import only"),
                                    (1.0, 0.5, "both (default)"),
                                    (1.0, 1.0, "both (equal)")]:
            result = evaluate_propagation(predictions, cc_index, imp_index,
                                           alpha=0.7, iterations=3,
                                           cc_weight=cc_w, imp_weight=imp_w)
            delta = result['R@1'] - baseline['R@1']
            print(f"{label:>20} {result['R@1']:>7.2f}% {result['R@5']:>7.2f}% {delta:>+7.2f}")


if __name__ == '__main__':
    main()
