"""
Multi-signal prediction expansion: combine co-change, import dependencies,
directory proximity, and test-source matching to boost recall.

Uses unified scoring where all signals contribute to a single score per candidate,
with multi-signal boost for files that appear across multiple signals.

Signals:
1. Co-change: files that historically change together (training data)
2. Import: files with import relationships (dep_graphs)
3. Directory: files in same directory as predicted files (file_trees)
4. Test-source: test/source file pairing heuristic
"""

import json
import argparse
import os
from collections import defaultdict, Counter
from typing import Dict, List, Set


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
                files = [ff for ff in item.get('changed_files', []) if ff.endswith('.py')]

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
    """Build per-repo bidirectional import index from dep_graphs."""
    index: Dict[str, Dict[str, Set[str]]] = {}

    if not os.path.isdir(dep_graph_dir):
        return index

    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith('_rels.json'):
            continue
        repo = fname.replace('_rels.json', '')

        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)

        neighbors: Dict[str, Set[str]] = defaultdict(set)

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


def build_dir_index(file_tree_dir: str) -> tuple:
    """Build per-repo directory->files index and all py files set."""
    dir_idx: Dict[str, Dict[str, List[str]]] = {}
    all_py: Dict[str, Set[str]] = {}

    if not os.path.isdir(file_tree_dir):
        return dir_idx, all_py

    for fname in os.listdir(file_tree_dir):
        if not fname.endswith('.json'):
            continue
        repo = fname.replace('.json', '')

        with open(os.path.join(file_tree_dir, fname)) as f:
            ft = json.load(f)

        dir_files: Dict[str, List[str]] = defaultdict(list)
        py_set = set(ft.get('py_files', []))
        all_py[repo] = py_set
        for py_file in py_set:
            d = os.path.dirname(py_file)
            dir_files[d].append(py_file)

        dir_idx[repo] = dict(dir_files)

    return dir_idx, all_py


def get_test_source_pairs(filepath: str, repo_files: Set[str]) -> List[str]:
    """Given a file path, return potential test/source counterparts."""
    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    pairs = []

    if basename.startswith('test_'):
        source_name = basename[5:]
        parent = os.path.dirname(dirname)
        pairs.append(os.path.join(dirname, source_name))
        pairs.append(os.path.join(parent, source_name))
        if os.path.basename(dirname) in ('tests', 'test'):
            parent2 = os.path.dirname(dirname)
            pairs.append(os.path.join(parent2, source_name))
    elif basename.endswith('_test.py'):
        source_name = basename[:-8] + '.py'
        pairs.append(os.path.join(dirname, source_name))
    else:
        pairs.append(os.path.join(dirname, 'test_' + basename))
        test_dir = os.path.join(dirname, 'tests')
        pairs.append(os.path.join(test_dir, 'test_' + basename))

    return [p for p in pairs if p in repo_files]


def expand_predictions(
    predictions_path: str,
    cochange_index: Dict,
    import_index: Dict,
    dir_index: Dict,
    all_py_files: Dict,
    output_path: str,
    max_expand: int = 35,
    min_cochange_score: float = 0.02,
    max_dir_size: int = 35,
    cc_weight: float = 1.0,
    import_weight: float = 0.6,
    dir_weight: float = 0.25,
    test_weight: float = 0.7,
    multi_signal_boost: float = 1.3,
) -> dict:
    """
    Unified multi-signal expansion.
    All candidates scored with weighted combination of signals.
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
        repo_files = all_py_files.get(repo, set())

        scores: Dict[str, float] = defaultdict(float)
        signal_count: Dict[str, int] = defaultdict(int)

        # 1. Co-change signal
        repo_cc = cochange_index.get(repo, {})
        for pred_file in original:
            for neighbor, score in repo_cc.get(pred_file, {}).items():
                if neighbor not in original_set and score >= min_cochange_score:
                    scores[neighbor] += cc_weight * score
                    signal_count[neighbor] += 1

        # 2. Import signal
        repo_imp = import_index.get(repo, {})
        for pred_file in original:
            for neighbor in repo_imp.get(pred_file, set()):
                if neighbor not in original_set:
                    scores[neighbor] += import_weight
                    signal_count[neighbor] += 1

        # 3. Directory proximity signal
        repo_dir = dir_index.get(repo, {})
        pred_dirs = set(os.path.dirname(f) for f in original)
        for d in pred_dirs:
            dir_files_list = repo_dir.get(d, [])
            if len(dir_files_list) > max_dir_size:
                continue
            specificity = 1.0 / max(len(dir_files_list), 1)
            for f in dir_files_list:
                if f not in original_set:
                    scores[f] += dir_weight * specificity
                    signal_count[f] += 1

        # 4. Test-source matching
        for pred_file in original:
            for pair in get_test_source_pairs(pred_file, repo_files):
                if pair not in original_set:
                    scores[pair] += test_weight
                    signal_count[pair] += 1

        # Multi-signal boost
        for f in list(scores.keys()):
            if signal_count[f] >= 2:
                scores[f] *= multi_signal_boost
            if signal_count[f] >= 3:
                scores[f] *= 1.2

        # Rank and select
        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        expansion = [c[0] for c in sorted_cands[:max_expand]]

        new_predicted = original + expansion

        # Recompute metrics
        gt = set(p.get('ground_truth', p.get('changed_py_files', [])))
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

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for ep in expanded_preds:
            f.write(json.dumps(ep) + '\n')

    # Summary
    summary = compute_summary(expanded_preds)
    summary_path = output_path.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    o = summary['overall']
    avg_exp = sum(p['num_expanded'] for p in expanded_preds) / len(expanded_preds)
    print(f"Avg expanded: {avg_exp:.1f}")
    print(f"Hit@1={o['hit@1']:.2f}% Hit@3={o['hit@3']:.2f}% Hit@5={o['hit@5']:.2f}% "
          f"Hit@10={o['hit@10']:.2f}% Hit@20={o['hit@20']:.2f}%")

    return summary


def compute_summary(preds: List[dict]) -> dict:
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
    parser.add_argument('--file_tree_dir', default='data/file_trees')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_expand', type=int, default=35)
    parser.add_argument('--min_cochange_score', type=float, default=0.02)
    parser.add_argument('--max_dir_size', type=int, default=35)
    parser.add_argument('--cc_weight', type=float, default=1.0)
    parser.add_argument('--import_weight', type=float, default=0.6)
    parser.add_argument('--dir_weight', type=float, default=0.25)
    parser.add_argument('--test_weight', type=float, default=0.7)
    args = parser.parse_args()

    print("Building co-change index...")
    cc_index = build_cochange_index(args.train_data, min_cochange=1)
    print(f"Co-change: {len(cc_index)} repos")

    print("Building import index...")
    imp_index = build_import_index(args.dep_graph_dir)
    print(f"Import: {len(imp_index)} repos")

    print("Building directory index...")
    dir_index, all_py = build_dir_index(args.file_tree_dir)
    print(f"Directory: {len(dir_index)} repos")

    print("Expanding predictions...")
    expand_predictions(
        args.predictions, cc_index, imp_index, dir_index, all_py, args.output,
        max_expand=args.max_expand,
        min_cochange_score=args.min_cochange_score,
        max_dir_size=args.max_dir_size,
        cc_weight=args.cc_weight,
        import_weight=args.import_weight,
        dir_weight=args.dir_weight,
        test_weight=args.test_weight,
    )


if __name__ == '__main__':
    main()
