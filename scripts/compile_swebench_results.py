#!/usr/bin/env python3
"""
Compile all SWE-bench results into a clean summary table.
Auto-discovers available neural reranking predictions.
"""
import os
import json
import re
from collections import defaultdict, Counter

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


def is_test_file(fp):
    parts = fp.split('/')
    for p in parts[:-1]:
        if p in ('test', 'tests', 'testing'):
            return True
    fn = parts[-1]
    return fn.startswith('test_') or fn.endswith('_test.py') or fn == 'conftest.py'


def postprocess(ranking, issue_text):
    # Deprioritize tests
    ranking = [f for f in ranking if not is_test_file(f)] + [f for f in ranking if is_test_file(f)]
    scores = {f: -i for i, f in enumerate(ranking)}
    N = len(ranking)
    # Traceback files
    tb_files = re.findall(r'File "([^"]+\.py)"', issue_text)
    file_refs = re.findall(r'[\w/]+\.py\b', issue_text)
    filenames = re.findall(r'\b(\w+\.py)\b', issue_text)
    for f in ranking:
        for tb_f in tb_files:
            if f.split('/')[-1] == tb_f.split('/')[-1]:
                scores[f] += N * 1.5
                break
        if any(f.endswith(p) for p in file_refs):
            scores[f] += N * 1.5
        elif any(f.endswith('/' + fn) or f == fn for fn in filenames):
            scores[f] += N * 0.7
        fn = f.split('/')[-1]
        if fn in ('setup.py', 'setup.cfg', 'manage.py'):
            scores[f] -= N * 0.3
    return sorted(ranking, key=lambda x: -scores[x])


def rrf_fusion(rankings, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] += w / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


def eval_metric(preds, gt_map, k_values=[1, 5, 10, 20]):
    """Return dict of metrics."""
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
    if count == 0:
        return {}
    result = {}
    for k in k_values:
        result[f'all@{k}'] = accs[k] / count * 100
        if matched_count > 0:
            result[f'274@{k}'] = matched_accs[k] / matched_count * 100
    result['n_all'] = count
    result['n_274'] = matched_count
    return result


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load GT and issue texts
    gt_map = {}
    issue_texts = {}
    test_path = os.path.join(base_dir, 'data/swebench_lite/swebench_lite_test.jsonl')
    with open(test_path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            gt_map[key] = set(d.get('changed_py_files', []))
            issue_texts[key] = d.get('issue_text', '')

    # Auto-discover neural predictions
    rerank_dirs = {}
    for model_dir in ['experiments/rankft_runB_graph', 'experiments/rankft_runA_bm25only',
                       'experiments/rankft_swebench_adapted']:
        full_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(full_path):
            continue
        for subdir in os.listdir(full_path):
            if subdir.startswith('eval_swebench'):
                pred_path = os.path.join(full_path, subdir, 'predictions.jsonl')
                if os.path.exists(pred_path):
                    model_name = os.path.basename(model_dir)
                    name = f"{model_name}/{subdir}"
                    rerank_dirs[name] = pred_path

    # Load neural predictions
    neural_preds = {}
    neural_gaps = {}
    print("=== Available Neural Reranking Results ===")
    for name, path in sorted(rerank_dirs.items()):
        preds = {}
        gaps = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                key = d.get('issue_id', d.get('instance_id', ''))
                preds[key] = d.get('predicted', d.get('reranked', []))
                raw_scores = d.get('scores', [])
                gaps[key] = raw_scores[0] - raw_scores[1] if len(raw_scores) >= 2 else 0
        neural_preds[name] = preds
        neural_gaps[name] = gaps
        # Evaluate
        metrics_raw = eval_metric(preds, gt_map)
        preds_pp = {k: postprocess(v, issue_texts.get(k, '')) for k, v in preds.items()}
        metrics_pp = eval_metric(preds_pp, gt_map)
        print(f"\n  {name} ({len(preds)} examples)")
        print(f"    Raw:  @1={metrics_raw.get('all@1',0):.1f}% @5={metrics_raw.get('all@5',0):.1f}% | 274@1={metrics_raw.get('274@1',0):.1f}%")
        print(f"    +PP:  @1={metrics_pp.get('all@1',0):.1f}% @5={metrics_pp.get('all@5',0):.1f}% | 274@1={metrics_pp.get('274@1',0):.1f}%")

    # Load BM25
    bm25_sources = {
        'bm25_tricked': os.path.join(base_dir, 'data/rankft/swebench_bm25_final_top500.jsonl'),
        'bm25_funcnotest': os.path.join(base_dir, 'data/rankft/swebench_bm25_function_notest_top500.jsonl'),
        'bm25_chunk': os.path.join(base_dir, 'data/rankft/swebench_bm25_chunk_top500.jsonl'),
    }
    bm25_preds = {}
    for name, path in bm25_sources.items():
        if os.path.exists(path):
            preds = {}
            for l in open(path):
                d = json.loads(l)
                key = d.get('issue_id', d.get('instance_id', ''))
                preds[key] = d.get('bm25_candidates', [])
            bm25_preds[name] = preds

    available_neural = list(neural_preds.keys())

    if len(available_neural) < 2:
        print("\nNeed at least 2 neural pools for fusion. Exiting.")
        return

    # Common keys
    all_keys = set(gt_map.keys())
    for name in available_neural:
        all_keys &= set(neural_preds[name].keys())
    for name in bm25_preds:
        all_keys &= set(bm25_preds[name].keys())
    all_keys = sorted(all_keys)

    # ============================================================
    # Best configurations
    # ============================================================
    print("\n\n" + "="*80)
    print("SUMMARY TABLE: Best SWE-bench Results")
    print("="*80)

    results = []

    # 1. BM25 baselines
    for bm25_name, pool in bm25_preds.items():
        m = eval_metric(pool, gt_map)
        results.append((f"BM25 {bm25_name}", m))

    # 2. Best single neural + PP
    best_neural_name = None
    best_neural_acc = 0
    for name in available_neural:
        preds_pp = {k: postprocess(neural_preds[name][k], issue_texts.get(k, ''))
                    for k in neural_preds[name]}
        m = eval_metric(preds_pp, gt_map)
        results.append((f"Neural {name} +PP", m))
        if m.get('274@1', 0) > best_neural_acc:
            best_neural_acc = m['274@1']
            best_neural_name = name

    # 3. Mega-fusion
    for rrf_k in [30]:
        fused = {}
        for key in all_keys:
            rankings = [postprocess(neural_preds[n][key], issue_texts.get(key, ''))
                       for n in available_neural if key in neural_preds[n]]
            for bn in bm25_preds:
                if key in bm25_preds[bn]:
                    rankings.append(bm25_preds[bn][key])
            fused[key] = rrf_fusion(rankings, k=rrf_k)
        fused_pp = {k: postprocess(v, issue_texts.get(k, '')) for k, v in fused.items()}
        m = eval_metric(fused_pp, gt_map)
        results.append((f"Mega-fusion RRF(k={rrf_k}) +PP", m))

    # 4. Majority vote
    for gap_threshold in [0.0, 0.5, 1.0]:
        fused = {}
        n_neural = 0
        for key in all_keys:
            if key not in bm25_preds.get('bm25_tricked', {}):
                continue
            confident_top1 = []
            confident_rankings = []
            for name in available_neural:
                if key not in neural_preds[name]:
                    continue
                gap = neural_gaps[name].get(key, 0)
                if gap >= gap_threshold:
                    r = postprocess(neural_preds[name][key], issue_texts.get(key, ''))
                    confident_top1.append(r[0])
                    confident_rankings.append(r)
            bm25_ranking = bm25_preds['bm25_tricked'][key]
            if confident_top1:
                top1_counts = Counter(confident_top1).most_common(1)
                consensus = top1_counts[0][0]
                agreement = top1_counts[0][1]
                if agreement >= 2:
                    seen = {consensus}
                    selected = [consensus]
                    for r in confident_rankings:
                        for f in r:
                            if f not in seen:
                                selected.append(f)
                                seen.add(f)
                    for f in bm25_ranking:
                        if f not in seen:
                            selected.append(f)
                            seen.add(f)
                    fused[key] = selected
                    n_neural += 1
                else:
                    fused[key] = bm25_ranking
            else:
                fused[key] = bm25_ranking
        m = eval_metric(fused, gt_map)
        results.append((f"MajVote(gap>={gap_threshold},agree>=2) n={n_neural}", m))

    # Print summary table
    print(f"\n{'Method':<55} {'All@1':>7} {'All@5':>7} {'All@10':>7} {'274@1':>7} {'274@5':>7}")
    print("-" * 95)
    for name, m in results:
        print(f"{name:<55} {m.get('all@1',0):>6.1f}% {m.get('all@5',0):>6.1f}% {m.get('all@10',0):>6.1f}% {m.get('274@1',0):>6.1f}% {m.get('274@5',0):>6.1f}%")

    # External baselines
    print("-" * 95)
    print(f"{'LocAgent BM25 (baseline, 274)':<55} {'':>7} {'':>7} {'':>7} {'38.7%':>7} {'':>7}")
    print(f"{'CodeRankEmbed (52.55%, 274)':<55} {'':>7} {'':>7} {'':>7} {'52.6%':>7} {'':>7}")

    # Oracle analysis
    print(f"\n{'--- Oracle Analysis ---'}")
    oracle_preds = {}
    for key in all_keys:
        gt = gt_map.get(key, set())
        if not gt:
            continue
        # Check if any neural pool is correct
        any_correct = False
        for name in available_neural:
            if key in neural_preds[name]:
                r = postprocess(neural_preds[name][key], issue_texts.get(key, ''))
                if gt.issubset(set(r[:1])):
                    any_correct = True
                    oracle_preds[key] = r
                    break
        if not any_correct and key in bm25_preds.get('bm25_tricked', {}):
            bm25_r = bm25_preds['bm25_tricked'][key]
            if gt.issubset(set(bm25_r[:1])):
                oracle_preds[key] = bm25_r
            else:
                oracle_preds[key] = bm25_r
    m = eval_metric(oracle_preds, gt_map)
    print(f"  Oracle (any neural PP or BM25): 274@1={m.get('274@1',0):.1f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
