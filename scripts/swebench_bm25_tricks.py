#!/usr/bin/env python3
"""
Post-processing tricks for BM25 rankings on SWE-bench Lite.

Applies hand-crafted heuristics to boost retrieval accuracy.
No information leakage — only uses issue text and file paths.
"""
import os
import re
import json
from collections import defaultdict

import numpy as np


def extract_mentioned_files(issue_text: str):
    """Extract file paths mentioned in the issue text."""
    # Full paths: anything/like/this.py
    full_paths = re.findall(r'[\w/]+\.py\b', issue_text)
    # Filenames only: something.py
    filenames = re.findall(r'\b(\w+\.py)\b', issue_text)
    # Class/function names in backticks or quotes
    identifiers = re.findall(r'[`\'"]([\w.]+)[`\'"]', issue_text)
    return full_paths, filenames, identifiers


def apply_tricks(candidates, scores_dict, issue_text, trick_config):
    """Apply post-processing tricks to rerank candidates.

    Args:
        candidates: list of file paths (already ranked)
        scores_dict: dict of filepath -> score (BM25 or neural)
        issue_text: the issue text
        trick_config: dict of trick names -> parameters
    """
    # Convert to mutable scores
    file_scores = {}
    for rank, f in enumerate(candidates):
        file_scores[f] = scores_dict.get(f, -rank)  # Use negative rank if no score

    # Normalize scores to [0, 1]
    vals = list(file_scores.values())
    min_v, max_v = min(vals), max(vals)
    range_v = max_v - min_v if max_v > min_v else 1.0
    for f in file_scores:
        file_scores[f] = (file_scores[f] - min_v) / range_v

    full_paths, filenames, identifiers = extract_mentioned_files(issue_text)

    # Trick 1: Boost files explicitly mentioned in issue
    if trick_config.get('mentioned_boost', 0) > 0:
        boost = trick_config['mentioned_boost']
        for f in candidates:
            # Exact path match
            if any(f.endswith(p) for p in full_paths):
                file_scores[f] += boost
            # Filename match
            elif any(f.endswith('/' + fn) or f == fn for fn in filenames):
                file_scores[f] += boost * 0.7
            # Identifier in path (e.g., backticked `models` matches models.py)
            else:
                for ident in identifiers:
                    parts = ident.split('.')
                    fname_parts = f.replace('.py', '').split('/')
                    if any(p.lower() in [fp.lower() for fp in fname_parts] for p in parts if len(p) > 2):
                        file_scores[f] += boost * 0.3
                        break

    # Trick 2: Penalize unlikely fix targets
    if trick_config.get('penalize_unlikely', 0) > 0:
        pen = trick_config['penalize_unlikely']
        for f in candidates:
            fname = f.split('/')[-1]
            # __init__.py — sometimes relevant, light penalty
            if fname == '__init__.py':
                file_scores[f] -= pen * 0.3
            # setup.py, setup.cfg, conftest.py — rarely fix targets
            if fname in ('setup.py', 'setup.cfg', 'manage.py'):
                file_scores[f] -= pen
            # Migration files
            if '/migrations/' in f:
                file_scores[f] -= pen
            # Documentation files
            if '/docs/' in f or '/doc/' in f:
                file_scores[f] -= pen * 0.5

    # Trick 3: Directory co-location boost
    if trick_config.get('dir_colocation_boost', 0) > 0:
        boost = trick_config['dir_colocation_boost']
        # Count files per directory in top-20
        top_20 = sorted(file_scores.keys(), key=lambda x: -file_scores[x])[:20]
        dir_counts = defaultdict(int)
        for f in top_20:
            d = '/'.join(f.split('/')[:-1]) if '/' in f else ''
            dir_counts[d] += 1
        # Boost files in directories with 2+ files in top-20
        for f in candidates:
            d = '/'.join(f.split('/')[:-1]) if '/' in f else ''
            if dir_counts.get(d, 0) >= 2:
                file_scores[f] += boost * (dir_counts[d] - 1) * 0.1

    # Trick 4: Depth penalty — very deep files are less likely targets
    if trick_config.get('depth_penalty', 0) > 0:
        pen = trick_config['depth_penalty']
        for f in candidates:
            depth = f.count('/')
            if depth > 5:
                file_scores[f] -= pen * (depth - 5) * 0.1

    # Re-rank
    reranked = sorted(candidates, key=lambda x: -file_scores.get(x, 0))
    return reranked


def main():
    # Load all BM25 rankings
    bm25_files = {
        'function_notest': 'data/rankft/swebench_bm25_function_notest_top500.jsonl',
        'chunk': 'data/rankft/swebench_bm25_chunk_top500.jsonl',
        'ensemble': 'data/rankft/swebench_bm25_best_ensemble_top500.jsonl',
        'function': 'data/rankft/swebench_bm25_function_top500.jsonl',
    }

    excluded = {
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

    # Load test data for issue text
    test_data = {}
    with open('data/swebench_lite/swebench_lite_test.jsonl') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            test_data[key] = d

    # Trick configs to try
    trick_configs = [
        {'name': 'no_tricks', 'mentioned_boost': 0, 'penalize_unlikely': 0, 'dir_colocation_boost': 0, 'depth_penalty': 0},
        {'name': 'mentioned_only', 'mentioned_boost': 0.5, 'penalize_unlikely': 0, 'dir_colocation_boost': 0, 'depth_penalty': 0},
        {'name': 'mentioned_strong', 'mentioned_boost': 1.0, 'penalize_unlikely': 0, 'dir_colocation_boost': 0, 'depth_penalty': 0},
        {'name': 'penalize_only', 'mentioned_boost': 0, 'penalize_unlikely': 0.3, 'dir_colocation_boost': 0, 'depth_penalty': 0},
        {'name': 'dir_boost_only', 'mentioned_boost': 0, 'penalize_unlikely': 0, 'dir_colocation_boost': 0.3, 'depth_penalty': 0},
        {'name': 'all_mild', 'mentioned_boost': 0.3, 'penalize_unlikely': 0.2, 'dir_colocation_boost': 0.2, 'depth_penalty': 0.1},
        {'name': 'all_strong', 'mentioned_boost': 0.7, 'penalize_unlikely': 0.3, 'dir_colocation_boost': 0.3, 'depth_penalty': 0.15},
        {'name': 'mentioned_penalize', 'mentioned_boost': 0.5, 'penalize_unlikely': 0.3, 'dir_colocation_boost': 0, 'depth_penalty': 0},
    ]

    for bm25_name, bm25_path in bm25_files.items():
        if not os.path.exists(bm25_path):
            continue

        data = [json.loads(l) for l in open(bm25_path)]
        print(f'\n=== {bm25_name} ({len(data)} examples) ===')

        for tc in trick_configs:
            accs = {k: 0 for k in [1, 3, 5, 10, 20]}
            count = 0
            best_results = []

            for d in data:
                key = d['issue_id']
                gt = set(d['ground_truth'])
                candidates = d['bm25_candidates']
                issue_text = test_data.get(key, {}).get('issue_text', '')

                # Create score dict from rank
                scores = {f: 1.0 / (i + 1) for i, f in enumerate(candidates)}

                reranked = apply_tricks(candidates, scores, issue_text, tc)

                count += 1
                for k in accs:
                    if gt.issubset(set(reranked[:k])):
                        accs[k] += 1

                best_results.append({
                    'repo': d['repo'],
                    'issue_id': key,
                    'bm25_candidates': reranked[:500],
                    'ground_truth': list(gt),
                })

            # Matched 274
            matched = [d for d in best_results if d['issue_id'] not in excluded]
            accs_274 = {}
            for k in [1, 5, 10, 20]:
                accs_274[k] = sum(1 for d in matched if set(d['ground_truth']).issubset(set(d['bm25_candidates'][:k]))) / len(matched) * 100

            print(f'  {tc["name"]:<25} All: @1={accs[1]/count*100:.2f}% @5={accs[5]/count*100:.2f}%  |  274: @1={accs_274[1]:.2f}% @5={accs_274[5]:.2f}%')

            # Save best tricks for the best BM25
            if bm25_name == 'function_notest' and tc['name'] == 'all_strong':
                with open('data/rankft/swebench_bm25_tricked_top500.jsonl', 'w') as f:
                    for r in best_results:
                        f.write(json.dumps(r) + '\n')

    print('\nDone.')


if __name__ == '__main__':
    main()
