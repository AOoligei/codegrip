#!/usr/bin/env python3
"""
Analyze all 1024-ctx reranking pools for SWE-bench Lite.
Evaluates individual pools, pairwise agreement, and multi-pool fusion strategies.
"""
import os
import json
import re
from collections import defaultdict, Counter

import numpy as np

np.random.seed(42)

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


def is_junk_file(f):
    fn = f.split('/')[-1]
    if fn == 'conf.py':
        return True
    if f.startswith('doc/') or f.startswith('docs/'):
        return True
    if 'makemigrations' in f:
        return True
    return False


def postprocess(ranking, issue_text):
    ranking = [f for f in ranking if not is_test_file(f)] + [f for f in ranking if is_test_file(f)]
    scores = {f: -i for i, f in enumerate(ranking)}
    N = len(ranking)
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


def load_preds(path):
    preds, scores = {}, {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            preds[key] = d.get('predicted', d.get('reranked', []))
            scores[key] = d.get('scores', [])
    return preds, scores


def eval_at_k(preds, gt_map, k_values=[1, 5, 10, 20]):
    accs = {k: 0 for k in k_values}
    count = 0
    for key, ranking in preds.items():
        if key in EXCLUDED_274:
            continue
        gt = gt_map.get(key, set())
        if not gt:
            continue
        count += 1
        for k in k_values:
            if gt.issubset(set(ranking[:k])):
                accs[k] += 1
    return {k: accs[k] / count * 100 if count > 0 else 0 for k in k_values}, count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # Load GT and issue texts
    gt_map = {}
    issue_texts = {}
    with open('data/swebench_lite/swebench_lite_test.jsonl') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            gt_map[key] = set(d.get('changed_py_files', []))
            issue_texts[key] = d.get('issue_text', '')

    # Load BM25 predictions
    bm25_preds = {}
    with open('data/rankft/swebench_bm25_final_top500.jsonl') as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            bm25_preds[key] = d.get('bm25_candidates', [])

    # Auto-discover 1024-ctx pools
    pool_dirs = {
        'best_ens_1024': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble_1024',
        'path_1024': 'experiments/rankft_runB_graph/eval_swebench_path_1024',
        'function_1024': 'experiments/rankft_runB_graph/eval_swebench_function_1024',
        'funcnotest_1024': 'experiments/rankft_runB_graph/eval_swebench_funcnotest_1024',
        'chunk_1024': 'experiments/rankft_runB_graph/eval_swebench_chunk_1024',
        'reformulated_1024': 'experiments/rankft_runB_graph/eval_swebench_reformulated_1024',
        'runA_path_1024': 'experiments/rankft_runA_bm25only/eval_swebench_path_1024',
    }

    # Also include 512-ctx pools for comparison
    pool_dirs_512 = {
        'best_ens_512': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble',
        'path_512': 'experiments/rankft_runB_graph/eval_swebench_500',
        'function_512': 'experiments/rankft_runB_graph/eval_swebench_function_bm25',
        'funcnotest_512': 'experiments/rankft_runB_graph/eval_swebench_funcnotest',
        'chunk_512': 'experiments/rankft_runB_graph/eval_swebench_chunk',
        'runA_ens_512': 'experiments/rankft_runA_bm25only/eval_swebench_best_ensemble',
    }

    # Load available pools
    all_preds = {}
    all_scores = {}
    print("=" * 80)
    print("1024-ctx Pool Evaluation")
    print("=" * 80)

    for name, path in sorted(pool_dirs.items()):
        pred_file = os.path.join(path, 'predictions.jsonl')
        if not os.path.exists(pred_file):
            print(f"  {name}: NOT READY")
            continue
        preds, scores = load_preds(pred_file)
        all_preds[name] = preds
        all_scores[name] = scores

        # Raw eval
        preds_pp = {k: postprocess(v, issue_texts.get(k, '')) for k, v in preds.items()}
        raw_m, n = eval_at_k(preds, gt_map)
        pp_m, _ = eval_at_k(preds_pp, gt_map)
        print(f"  {name}: raw @1={raw_m[1]:.1f}% pp @1={pp_m[1]:.1f}% @5={pp_m[5]:.1f}% @10={pp_m[10]:.1f}% (n={n})")

    for name, path in sorted(pool_dirs_512.items()):
        pred_file = os.path.join(path, 'predictions.jsonl')
        if not os.path.exists(pred_file):
            continue
        preds, scores = load_preds(pred_file)
        all_preds[name] = preds
        all_scores[name] = scores

    if len([n for n in all_preds if '1024' in n]) < 2:
        print("\nNeed at least 2 1024-ctx pools for fusion analysis. Exiting.")
        return

    # Common keys (274 only)
    keys_274 = sorted(k for k in gt_map if k not in EXCLUDED_274 and gt_map.get(k))
    keys_274 = [k for k in keys_274 if all(k in all_preds[n] for n in all_preds) and k in bm25_preds]

    pool_1024_names = sorted(n for n in all_preds if '1024' in n)
    print(f"\nAvailable 1024-ctx pools: {pool_1024_names}")
    print(f"Common keys: {len(keys_274)}")

    # ============================================================
    # Section 2: Confidence switching per pool
    # ============================================================
    print("\n" + "=" * 80)
    print("Confidence Switching (gap >= threshold → neural, else → BM25)")
    print("=" * 80)

    for name in pool_1024_names:
        scores = all_scores[name]
        preds = all_preds[name]
        for gap_thresh in [0.5, 1.0, 1.5, 2.0]:
            correct = 0
            n_neural = 0
            for key in keys_274:
                r = postprocess(preds[key], issue_texts.get(key, ''))
                sc = scores.get(key, [])
                gap = sc[0] - sc[1] if len(sc) >= 2 else 0
                top1 = r[0] if r else ''
                if is_junk_file(top1):
                    fused = bm25_preds[key]
                elif gap >= gap_thresh:
                    fused = r
                    n_neural += 1
                else:
                    fused = bm25_preds[key]
                if gt_map[key].issubset(set(fused[:1])):
                    correct += 1
            print(f"  {name} gap>={gap_thresh}: neural={n_neural} → @1={correct/len(keys_274)*100:.1f}%")

    # ============================================================
    # Section 3: Multi-pool Majority Vote (1024-ctx only)
    # ============================================================
    print("\n" + "=" * 80)
    print("Multi-Pool Majority Vote (1024-ctx pools)")
    print("=" * 80)

    for gap_thresh in [0.0, 0.5, 1.0]:
        for min_agree in [2, 3]:
            correct = 0
            n_neural = 0
            for key in keys_274:
                top1_votes = []
                confident_rankings = []
                for name in pool_1024_names:
                    sc = all_scores[name].get(key, [])
                    gap = sc[0] - sc[1] if len(sc) >= 2 else 0
                    if gap >= gap_thresh:
                        r = postprocess(all_preds[name][key], issue_texts.get(key, ''))
                        top1 = r[0] if r else ''
                        if not is_junk_file(top1):
                            top1_votes.append(top1)
                            confident_rankings.append(r)

                if top1_votes:
                    counts = Counter(top1_votes).most_common(1)
                    consensus = counts[0][0]
                    agreement = counts[0][1]
                    if agreement >= min_agree:
                        seen = {consensus}
                        selected = [consensus]
                        for r in confident_rankings:
                            for f in r:
                                if f not in seen:
                                    selected.append(f)
                                    seen.add(f)
                        for f in bm25_preds[key]:
                            if f not in seen:
                                selected.append(f)
                                seen.add(f)
                        fused = selected
                        n_neural += 1
                    else:
                        fused = bm25_preds[key]
                else:
                    fused = bm25_preds[key]

                if gt_map[key].issubset(set(fused[:1])):
                    correct += 1
            print(f"  gap>={gap_thresh} agree>={min_agree}: neural={n_neural} → @1={correct/len(keys_274)*100:.1f}%")

    # ============================================================
    # Section 4: RRF Fusion (1024-ctx pools + BM25)
    # ============================================================
    print("\n" + "=" * 80)
    print("RRF Fusion (1024-ctx pools + BM25)")
    print("=" * 80)

    for rrf_k in [30, 60]:
        for include_bm25 in [False, True]:
            correct = 0
            for key in keys_274:
                rankings = []
                for name in pool_1024_names:
                    r = postprocess(all_preds[name][key], issue_texts.get(key, ''))
                    rankings.append(r)
                if include_bm25:
                    rankings.append(bm25_preds[key])
                fused = rrf_fusion(rankings, k=rrf_k)
                if gt_map[key].issubset(set(fused[:1])):
                    correct += 1
            bm25_str = "+BM25" if include_bm25 else ""
            print(f"  RRF(k={rrf_k}) {len(pool_1024_names)} pools{bm25_str}: @1={correct/len(keys_274)*100:.1f}%")

    # ============================================================
    # Section 5: Best single pool cascade + multi-pool MV fallback
    # ============================================================
    print("\n" + "=" * 80)
    print("Cascade: Best Pool Confidence → Multi-Pool MV → BM25")
    print("=" * 80)

    # Find best single 1024-ctx pool
    best_pool = None
    best_acc = 0
    for name in pool_1024_names:
        preds = all_preds[name]
        preds_pp = {k: postprocess(v, issue_texts.get(k, '')) for k, v in preds.items()}
        m, _ = eval_at_k(preds_pp, gt_map)
        if m[1] > best_acc:
            best_acc = m[1]
            best_pool = name
    print(f"  Best single pool: {best_pool} ({best_acc:.1f}%)")

    for gap_best in [1.0, 1.5, 2.0]:
        for gap_mv in [0.0, 0.5]:
            for min_agree in [2]:
                correct = 0
                n_best = n_mv = n_bm25 = 0
                for key in keys_274:
                    # Level 1: best pool confident
                    sc = all_scores[best_pool].get(key, [])
                    gap = sc[0] - sc[1] if len(sc) >= 2 else 0
                    r_best = postprocess(all_preds[best_pool][key], issue_texts.get(key, ''))
                    top1 = r_best[0] if r_best else ''

                    if not is_junk_file(top1) and gap >= gap_best:
                        fused = r_best
                        n_best += 1
                    else:
                        # Level 2: multi-pool MV
                        top1_votes = []
                        for name in pool_1024_names:
                            sc2 = all_scores[name].get(key, [])
                            gap2 = sc2[0] - sc2[1] if len(sc2) >= 2 else 0
                            if gap2 >= gap_mv:
                                r2 = postprocess(all_preds[name][key], issue_texts.get(key, ''))
                                t = r2[0] if r2 else ''
                                if not is_junk_file(t):
                                    top1_votes.append(t)
                        if top1_votes:
                            counts = Counter(top1_votes).most_common(1)
                            if counts[0][1] >= min_agree:
                                fused = [counts[0][0]] + [f for f in bm25_preds[key] if f != counts[0][0]]
                                n_mv += 1
                            else:
                                fused = bm25_preds[key]
                                n_bm25 += 1
                        else:
                            fused = bm25_preds[key]
                            n_bm25 += 1

                    if gt_map[key].issubset(set(fused[:1])):
                        correct += 1
                print(f"  best>={gap_best} mv_gap>={gap_mv} agree>={min_agree}: best={n_best} mv={n_mv} bm25={n_bm25} → @1={correct/len(keys_274)*100:.1f}%")

    # ============================================================
    # Section 6: Oracle analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("Oracle Analysis")
    print("=" * 80)

    oracle_correct = 0
    for key in keys_274:
        gt = gt_map[key]
        any_correct = False
        for name in pool_1024_names:
            r = postprocess(all_preds[name][key], issue_texts.get(key, ''))
            if gt.issubset(set(r[:1])):
                any_correct = True
                break
        if not any_correct:
            if gt.issubset(set(bm25_preds[key][:1])):
                any_correct = True
        if any_correct:
            oracle_correct += 1
    print(f"  Oracle (any 1024-ctx pool PP or BM25 @1): {oracle_correct/len(keys_274)*100:.1f}%")

    # Pairwise diversity
    print("\n  Pairwise agreement (top-1 after PP):")
    for i, n1 in enumerate(pool_1024_names):
        for n2 in pool_1024_names[i+1:]:
            agree = 0
            for key in keys_274:
                r1 = postprocess(all_preds[n1][key], issue_texts.get(key, ''))
                r2 = postprocess(all_preds[n2][key], issue_texts.get(key, ''))
                if r1 and r2 and r1[0] == r2[0]:
                    agree += 1
            print(f"    {n1} vs {n2}: {agree/len(keys_274)*100:.1f}% agree")

    print("\nDone.")


if __name__ == '__main__':
    main()
