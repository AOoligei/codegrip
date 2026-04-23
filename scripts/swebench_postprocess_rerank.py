#!/usr/bin/env python3
"""
Post-process SWE-bench neural reranking predictions.

Applies test file exclusion and other tricks to already-completed
reranking results without needing GPU.

Key trick: deprioritize test files in neural rankings (move to bottom).
This is the single biggest factor (+9% for BM25, should help neural too).
"""
import os
import json
import re
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


def is_test_file(filepath):
    parts = filepath.split('/')
    for part in parts[:-1]:
        if part in ('test', 'tests', 'testing'):
            return True
    fname = parts[-1]
    if fname.startswith('test_') or fname.endswith('_test.py'):
        return True
    if fname == 'conftest.py':
        return True
    return False


def deprioritize_tests(ranking):
    """Move test files to the bottom of the ranking."""
    non_test = [f for f in ranking if not is_test_file(f)]
    test = [f for f in ranking if is_test_file(f)]
    return non_test + test


def extract_traceback_files(issue_text):
    """Extract file paths from Python tracebacks in issue text."""
    # Match: File "path/to/file.py", line N
    tb_files = re.findall(r'File "([^"]+\.py)"', issue_text)
    # Normalize: extract relative paths from absolute ones
    normalized = []
    for f in tb_files:
        parts = f.split('/')
        # Find where the repo source starts (skip site-packages, lib, etc.)
        for i, part in enumerate(parts):
            if part in ('site-packages', 'lib', 'dist-packages'):
                # Take everything after the package name
                if i + 1 < len(parts):
                    normalized.append('/'.join(parts[i+1:]))
                break
        else:
            # No site-packages, just use as-is or last N components
            normalized.append(f)
    return normalized


def extract_module_paths(issue_text):
    """Extract Python module references and convert to file paths."""
    # Dotted module references like django.db.models
    modules = re.findall(r'\b([a-z][a-z0-9_]+(?:\.[a-z][a-z0-9_]+){1,})\b', issue_text)
    paths = set()
    for m in modules:
        # Skip common non-module patterns
        if any(m.startswith(p) for p in ('e.g', 'i.e', 'http', 'www', 'v1', 'v2')):
            continue
        parts = m.split('.')
        # module.path -> module/path.py
        paths.add('/'.join(parts) + '.py')
        # Also without last component (class or function name)
        if len(parts) > 2:
            paths.add('/'.join(parts[:-1]) + '.py')
    return paths


def apply_heuristic_boost(ranking, issue_text):
    """Boost files mentioned in issue, penalize unlikely targets."""
    scores = {f: -i for i, f in enumerate(ranking)}
    N = len(ranking)

    # 1. Traceback file extraction — strongest signal
    tb_files = extract_traceback_files(issue_text)
    for f in ranking:
        for tb_f in tb_files:
            # Match by suffix (handles path prefix differences)
            if f.endswith(tb_f) or tb_f.endswith(f):
                scores[f] += N * 2  # Very strong boost
                break
            # Match by filename only
            elif f.split('/')[-1] == tb_f.split('/')[-1]:
                scores[f] += N * 0.8
                break

    # 2. Explicit file path mentions
    file_refs = re.findall(r'[\w/]+\.py\b', issue_text)
    filenames = re.findall(r'\b(\w+\.py)\b', issue_text)

    for f in ranking:
        if any(f.endswith(p) for p in file_refs):
            scores[f] += N * 1.5  # Strong boost
        elif any(f.endswith('/' + fn) or f == fn for fn in filenames):
            scores[f] += N * 0.7

    # 3. Module reference matching
    module_paths = extract_module_paths(issue_text)
    for f in ranking:
        for mp in module_paths:
            if f.endswith(mp) or mp.endswith(f):
                scores[f] += N * 0.5
                break

    # 4. Backticked identifier matching
    identifiers = re.findall(r'`([\w.]+)`', issue_text)
    for f in ranking:
        fname_parts = f.replace('.py', '').split('/')
        for ident in identifiers:
            parts = ident.split('.')
            if any(p.lower() in [fp.lower() for fp in fname_parts]
                   for p in parts if len(p) > 2):
                scores[f] += N * 0.2
                break

    # 5. Penalize unlikely fix targets
    for f in ranking:
        fname = f.split('/')[-1]
        if fname in ('setup.py', 'setup.cfg', 'manage.py'):
            scores[f] -= N * 0.3
        if fname == '__init__.py':
            scores[f] -= N * 0.1
        if '/migrations/' in f:
            scores[f] -= N * 0.3
        if '/docs/' in f or '/doc/' in f:
            scores[f] -= N * 0.2

    return sorted(ranking, key=lambda x: -scores[x])


def evaluate(preds, gt_map, k_values=[1, 3, 5, 10, 20], label=""):
    accs = {k: 0 for k in k_values}
    matched_accs = {k: 0 for k in k_values}
    count = 0
    matched_count = 0

    for key, ranking in preds.items():
        gt = gt_map.get(key, set())
        if not gt:
            continue
        count += 1
        is_matched = key not in EXCLUDED_274
        if is_matched:
            matched_count += 1

        for k in k_values:
            if gt.issubset(set(ranking[:k])):
                accs[k] += 1
                if is_matched:
                    matched_accs[k] += 1

    all_str = " ".join(f"@{k}={accs[k]/count*100:.2f}%" for k in k_values)
    m_str = " ".join(f"@{k}={matched_accs[k]/matched_count*100:.2f}%" for k in [1, 5]) if matched_count > 0 else ""
    print(f"  {label:<30} All({count}): {all_str}  |  274({matched_count}): {m_str}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load GT
    gt_map = {}
    issue_texts = {}
    test_path = os.path.join(base_dir, 'data/swebench_lite/swebench_lite_test.jsonl')
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            gt_map[key] = set(d.get('changed_py_files', []))
            issue_texts[key] = d.get('issue_text', '')

    # Find all completed SWE-bench reranking predictions
    rerank_dirs = {
        'path_bm25': 'experiments/rankft_runB_graph/eval_swebench_500',
        'content_bm25': 'experiments/rankft_runB_graph/eval_swebench_content_bm25',
        'function_bm25': 'experiments/rankft_runB_graph/eval_swebench_function_bm25',
        'funcnotest': 'experiments/rankft_runB_graph/eval_swebench_funcnotest',
        'chunk': 'experiments/rankft_runB_graph/eval_swebench_chunk',
        'best_ensemble': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble',
        'best_ens_1024': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble_1024',
        'runA_best_ens': 'experiments/rankft_runA_bm25only/eval_swebench_best_ensemble',
    }

    for name, dir_path in rerank_dirs.items():
        pred_path = os.path.join(base_dir, dir_path, 'predictions.jsonl')
        if not os.path.exists(pred_path):
            continue

        print(f"\n=== {name} ===")
        preds = {}
        with open(pred_path) as f:
            for line in f:
                d = json.loads(line)
                key = d.get('issue_id', d.get('instance_id', ''))
                preds[key] = d.get('predicted', d.get('reranked', []))

        # Original
        evaluate(preds, gt_map, label="original")

        # Test deprioritized
        preds_notest = {k: deprioritize_tests(v) for k, v in preds.items()}
        evaluate(preds_notest, gt_map, label="deprioritize tests")

        # With heuristic boost
        preds_boosted = {k: apply_heuristic_boost(v, issue_texts.get(k, '')) for k, v in preds.items()}
        evaluate(preds_boosted, gt_map, label="heuristic boost")

        # Both
        preds_both = {k: apply_heuristic_boost(deprioritize_tests(v), issue_texts.get(k, ''))
                      for k, v in preds.items()}
        evaluate(preds_both, gt_map, label="tests + boost")

    print("\nDone.")


if __name__ == '__main__':
    main()
