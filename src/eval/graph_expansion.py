"""
Graph-based prediction expansion: enhance SFT predictions using
import dependencies + co-change patterns from training data.

Two expansion strategies:
1. Co-change expansion: files that historically change together
2. Import expansion: files that import/are imported by predicted files
"""

import json
import argparse
import os
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def build_cochange_index(train_data_path: str, min_cochange: int = 1) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build per-repo co-change index from training PR data."""
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

            for f_item in files:
                repo_file_count[repo][f_item] += 1

            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for repo in repo_cochanges:
        index[repo] = defaultdict(dict)
        for (fa, fb), count in repo_cochanges[repo].items():
            if count >= min_cochange:
                score = count / max(repo_file_count[repo][fa], 1)
                index[repo][fa][fb] = score

    return index


def build_import_index(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build per-repo bidirectional import index.
    Returns: {repo: {file: set of related files (imports + imported-by)}}
    """
    index: Dict[str, Dict[str, Set[str]]] = {}

    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith('_rels.json'):
            continue
        repo = fname.replace('_rels.json', '')

        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)

        file_imports = rels.get('file_imports', {})
        # Build bidirectional index
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for src, targets in file_imports.items():
            for tgt in targets:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)  # bidirectional

        # Also add call-graph based file-level connections
        call_graph = rels.get('call_graph', {})
        for src_func, callees in call_graph.items():
            src_file = src_func.split(':')[0] if ':' in src_func else src_func
            for callee in callees:
                tgt_file = callee.split(':')[0] if ':' in callee else callee
                if src_file != tgt_file:
                    neighbors[src_file].add(tgt_file)
                    neighbors[tgt_file].add(src_file)

        index[repo] = dict(neighbors)

    return index


def expand_predictions(
    predictions_path: str,
    cochange_index: Dict[str, Dict[str, Dict[str, float]]],
    import_index: Dict[str, Dict[str, Set[str]]],
    output_path: str,
    max_cochange: int = 10,
    max_import: int = 10,
    min_cochange_score: float = 0.05,
) -> None:
    """
    Expand SFT predictions using co-change + import-based neighbors.
    Order: original predictions -> co-change expansions -> import expansions
    """
    preds = []
    with open(predictions_path) as f:
        for line in f:
            preds.append(json.loads(line))

    expanded_preds = []

    for p in preds:
        repo = p['repo']
        original = list(p['predicted'])
        original_set = set(original)

        # 1. Co-change expansion
        cochange_cands: Dict[str, float] = {}
        repo_cc = cochange_index.get(repo, {})
        for pred_file in original:
            neighbors = repo_cc.get(pred_file, {})
            for neighbor, score in neighbors.items():
                if neighbor not in original_set and score >= min_cochange_score:
                    cochange_cands[neighbor] = max(cochange_cands.get(neighbor, 0), score)

        sorted_cc = sorted(cochange_cands.items(), key=lambda x: x[1], reverse=True)
        cc_expansion = [c[0] for c in sorted_cc[:max_cochange]]

        # 2. Import expansion (for files not already covered)
        expanded_so_far = original_set | set(cc_expansion)
        import_cands: List[str] = []
        repo_imp = import_index.get(repo, {})
        # Score imports by how many predicted files link to them
        import_scores: Dict[str, int] = defaultdict(int)
        for pred_file in original:
            for neighbor in repo_imp.get(pred_file, set()):
                if neighbor not in expanded_so_far:
                    import_scores[neighbor] += 1

        sorted_imp = sorted(import_scores.items(), key=lambda x: x[1], reverse=True)
        imp_expansion = [c[0] for c in sorted_imp[:max_import]]

        new_predicted = original + cc_expansion + imp_expansion

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
        new_p['num_cochange_expanded'] = len(cc_expansion)
        new_p['num_import_expanded'] = len(imp_expansion)
        expanded_preds.append(new_p)

    # Write
    with open(output_path, 'w') as f:
        for p in expanded_preds:
            f.write(json.dumps(p) + '\n')

    # Summary
    summary = compute_summary(expanded_preds)
    summary_path = output_path.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    o = summary['overall']
    avg_cc = sum(p['num_cochange_expanded'] for p in expanded_preds) / len(expanded_preds)
    avg_imp = sum(p['num_import_expanded'] for p in expanded_preds) / len(expanded_preds)
    print(f"Avg expansions: co-change={avg_cc:.1f}, import={avg_imp:.1f}")
    print(f"Hit@1={o['hit@1']:.2f}% Hit@3={o['hit@3']:.2f}% Hit@5={o['hit@5']:.2f}% "
          f"Hit@10={o['hit@10']:.2f}% Hit@20={o['hit@20']:.2f}%")


def compute_summary(preds: List[dict]) -> dict:
    from collections import defaultdict
    repo_metrics = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)
    for p in preds:
        repo = p['repo']
        for k, v in p['metrics'].items():
            repo_metrics[repo][k].append(v)
            overall[k].append(v)

    summary = {
        'overall': {k: sum(v) / len(v) for k, v in overall.items()},
        'per_repo': {}
    }
    for repo, metrics in repo_metrics.items():
        summary['per_repo'][repo] = {}
        for k, v in metrics.items():
            summary['per_repo'][repo][k] = sum(v) / len(v)
        summary['per_repo'][repo]['count'] = len(repo_metrics[repo].get('hit@1', []))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_cochange', type=int, default=10)
    parser.add_argument('--max_import', type=int, default=10)
    parser.add_argument('--min_cochange_score', type=float, default=0.05)
    args = parser.parse_args()

    print("Building co-change index...")
    cc_index = build_cochange_index(args.train_data, min_cochange=1)
    print(f"Co-change: {len(cc_index)} repos")

    print("Building import index...")
    imp_index = build_import_index(args.dep_graph_dir)
    print(f"Import: {len(imp_index)} repos")

    print("Expanding predictions...")
    expand_predictions(
        args.predictions, cc_index, imp_index, args.output,
        max_cochange=args.max_cochange,
        max_import=args.max_import,
        min_cochange_score=args.min_cochange_score,
    )


if __name__ == '__main__':
    main()
