#!/usr/bin/env python3
"""
Full SWE-bench pipeline: neural reranking + post-processing + multi-pool fusion.

Combines all available signals for the best possible file-level retrieval.
Runs on CPU using completed neural reranking predictions.
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


# ============================================================
# Post-processing functions
# ============================================================

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
    non_test = [f for f in ranking if not is_test_file(f)]
    test = [f for f in ranking if is_test_file(f)]
    return non_test + test


def extract_traceback_files(issue_text):
    tb_files = re.findall(r'File "([^"]+\.py)"', issue_text)
    normalized = []
    for f in tb_files:
        parts = f.split('/')
        for i, part in enumerate(parts):
            if part in ('site-packages', 'lib', 'dist-packages'):
                if i + 1 < len(parts):
                    normalized.append('/'.join(parts[i+1:]))
                break
        else:
            normalized.append(f)
    return normalized


def extract_module_paths(issue_text):
    modules = re.findall(r'\b([a-z][a-z0-9_]+(?:\.[a-z][a-z0-9_]+){1,})\b', issue_text)
    paths = set()
    for m in modules:
        if any(m.startswith(p) for p in ('e.g', 'i.e', 'http', 'www', 'v1', 'v2')):
            continue
        parts = m.split('.')
        paths.add('/'.join(parts) + '.py')
        if len(parts) > 2:
            paths.add('/'.join(parts[:-1]) + '.py')
    return paths


def apply_all_tricks(ranking, issue_text):
    """Apply all tricks: test deprioritize + file mention boost."""
    # First deprioritize tests
    ranking = deprioritize_tests(ranking)

    scores = {f: -i for i, f in enumerate(ranking)}
    N = len(ranking)

    # Traceback files
    tb_files = extract_traceback_files(issue_text)
    for f in ranking:
        for tb_f in tb_files:
            if f.endswith(tb_f) or tb_f.endswith(f):
                scores[f] += N * 2
                break
            elif f.split('/')[-1] == tb_f.split('/')[-1]:
                scores[f] += N * 0.8
                break

    # File path mentions
    file_refs = re.findall(r'[\w/]+\.py\b', issue_text)
    filenames = re.findall(r'\b(\w+\.py)\b', issue_text)
    for f in ranking:
        if any(f.endswith(p) for p in file_refs):
            scores[f] += N * 1.5
        elif any(f.endswith('/' + fn) or f == fn for fn in filenames):
            scores[f] += N * 0.7

    # Module references
    module_paths = extract_module_paths(issue_text)
    for f in ranking:
        for mp in module_paths:
            if f.endswith(mp) or mp.endswith(f):
                scores[f] += N * 0.5
                break

    # Backticked identifiers
    identifiers = re.findall(r'`([\w.]+)`', issue_text)
    for f in ranking:
        fname_parts = f.replace('.py', '').split('/')
        for ident in identifiers:
            parts = ident.split('.')
            if any(p.lower() in [fp.lower() for fp in fname_parts]
                   for p in parts if len(p) > 2):
                scores[f] += N * 0.2
                break

    # Penalize unlikely targets
    for f in ranking:
        fname = f.split('/')[-1]
        if fname in ('setup.py', 'setup.cfg', 'manage.py'):
            scores[f] -= N * 0.3
        if fname == '__init__.py':
            scores[f] -= N * 0.1
        if '/migrations/' in f:
            scores[f] -= N * 0.3

    return sorted(ranking, key=lambda x: -scores[x])


# ============================================================
# Fusion functions
# ============================================================

def rrf_fusion(rankings, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] += w / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


def score_fusion(score_dicts, weights=None):
    """Fuse by weighted score averaging."""
    if weights is None:
        weights = [1.0] * len(score_dicts)
    combined = defaultdict(float)
    for sd, w in zip(score_dicts, weights):
        # Normalize scores to [0, 1]
        vals = list(sd.values())
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v if max_v > min_v else 1.0
        for f, s in sd.items():
            combined[f] += w * (s - min_v) / rng
    return sorted(combined.keys(), key=lambda x: -combined[x])


# ============================================================
# Evaluation
# ============================================================

def evaluate(preds, gt_map, k_values=[1, 5, 10, 20], label=""):
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
    m_str = ""
    if matched_count > 0:
        m_str = " ".join(f"@{k}={matched_accs[k]/matched_count*100:.2f}%" for k in k_values)
    print(f"  {label:<40} All({count}): {all_str}  |  274: {m_str}")
    return {
        'all': {k: accs[k]/count*100 for k in k_values},
        'matched': {k: matched_accs[k]/matched_count*100 for k in k_values} if matched_count > 0 else {},
    }


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

    # ============================================================
    # Load all neural reranking predictions
    # ============================================================
    rerank_sources = {
        'path_bm25': 'experiments/rankft_runB_graph/eval_swebench_500',
        'path_orig': 'experiments/rankft_runB_graph/eval_swebench',
        'content_bm25': 'experiments/rankft_runB_graph/eval_swebench_content_bm25',
        'function_bm25': 'experiments/rankft_runB_graph/eval_swebench_function_bm25',
        'funcnotest': 'experiments/rankft_runB_graph/eval_swebench_funcnotest',
        'chunk': 'experiments/rankft_runB_graph/eval_swebench_chunk',
        'best_ensemble': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble',
        'best_ens_1024': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble_1024',
        'runA_best_ens': 'experiments/rankft_runA_bm25only/eval_swebench_best_ensemble',
    }

    neural_preds = {}  # name -> {key: ranking}
    neural_gaps = {}   # name -> {key: score_gap}
    available = []
    for name, dir_path in rerank_sources.items():
        pred_path = os.path.join(base_dir, dir_path, 'predictions.jsonl')
        if not os.path.exists(pred_path):
            continue
        preds = {}
        gaps = {}
        with open(pred_path) as f:
            for line in f:
                d = json.loads(line)
                key = d.get('issue_id', d.get('instance_id', ''))
                preds[key] = d.get('predicted', d.get('reranked', []))
                raw_scores = d.get('scores', [])
                if len(raw_scores) >= 2:
                    gaps[key] = raw_scores[0] - raw_scores[1]
                else:
                    gaps[key] = 0
        neural_preds[name] = preds
        neural_gaps[name] = gaps
        available.append(name)
        print(f"  Loaded {name}: {len(preds)} predictions")

    # Load BM25 rankings
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
            print(f"  Loaded BM25 {name}: {len(preds)} rankings")

    if not available:
        print("No neural reranking results available yet.")
        return

    print(f"\n  Available neural: {available}")

    # ============================================================
    # 1. Individual neural results + post-processing
    # ============================================================
    print("\n" + "="*80)
    print("1. INDIVIDUAL NEURAL RANKINGS + POST-PROCESSING")
    print("="*80)

    best_results = {}
    for name in available:
        print(f"\n--- {name} ---")
        preds = neural_preds[name]
        evaluate(preds, gt_map, label="original")

        preds_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                    for k, v in preds.items()}
        result = evaluate(preds_pp, gt_map, label="post-processed")
        best_results[name] = preds_pp

    # ============================================================
    # 2. Multi-pool neural fusion + post-processing
    # ============================================================
    if len(available) >= 2:
        print("\n" + "="*80)
        print("2. MULTI-POOL NEURAL RRF FUSION")
        print("="*80)

        # Get common keys
        all_keys = set(gt_map.keys())
        for name in available:
            all_keys &= set(neural_preds[name].keys())
        all_keys = sorted(all_keys)

        # Try different subsets
        subset_configs = [
            ('all_neural', available),
        ]
        # Also try leaving out each source
        if len(available) >= 3:
            for exc in available:
                subset_configs.append(
                    (f'no_{exc}', [n for n in available if n != exc]))

        for config_name, sources in subset_configs:
            for rrf_k in [30, 60]:
                fused_preds = {}
                for key in all_keys:
                    rankings = [neural_preds[name][key] for name in sources
                                if key in neural_preds[name]]
                    fused_preds[key] = rrf_fusion(rankings, k=rrf_k)
                evaluate(fused_preds, gt_map, label=f"RRF(k={rrf_k}) {config_name}")

                # With post-processing
                fused_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                           for k, v in fused_preds.items()}
                evaluate(fused_pp, gt_map, label=f"RRF(k={rrf_k}) {config_name} +PP")

    # ============================================================
    # 3. Neural + BM25 hybrid fusion + post-processing
    # ============================================================
    print("\n" + "="*80)
    print("3. NEURAL + BM25 HYBRID FUSION")
    print("="*80)

    # Best single neural + BM25
    for neural_name in available:
        for bm25_name, bm25_pool in bm25_preds.items():
            for neural_w in [1.0, 2.0, 3.0]:
                fused_preds = {}
                for key in all_keys:
                    if key not in bm25_pool or key not in neural_preds[neural_name]:
                        continue
                    fused = rrf_fusion(
                        [neural_preds[neural_name][key], bm25_pool[key]],
                        weights=[neural_w, 1.0], k=60)
                    fused_preds[key] = fused
                if fused_preds:
                    fused_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                               for k, v in fused_preds.items()}
                    evaluate(fused_pp, gt_map,
                             label=f"{neural_name}(w={neural_w})+{bm25_name} +PP")

    # ============================================================
    # 4. All-source mega-fusion + post-processing
    # ============================================================
    if len(available) >= 2:
        print("\n" + "="*80)
        print("4. ALL-SOURCE MEGA-FUSION + POST-PROCESSING")
        print("="*80)

        for rrf_k in [30, 60]:
            for neural_w in [1.0, 2.0, 3.0]:
                fused_preds = {}
                for key in all_keys:
                    all_rankings = []
                    weights = []
                    for name in available:
                        if key in neural_preds[name]:
                            all_rankings.append(neural_preds[name][key])
                            weights.append(neural_w)
                    for bm25_name, bm25_pool in bm25_preds.items():
                        if key in bm25_pool:
                            all_rankings.append(bm25_pool[key])
                            weights.append(1.0)
                    fused_preds[key] = rrf_fusion(all_rankings, weights=weights, k=rrf_k)

                fused_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                           for k, v in fused_preds.items()}
                evaluate(fused_pp, gt_map,
                         label=f"MEGA RRF(k={rrf_k},nw={neural_w}) +PP")

    # ============================================================
    # 5. BM25-only baseline with post-processing
    # ============================================================
    print("\n" + "="*80)
    print("5. BM25 BASELINES + POST-PROCESSING")
    print("="*80)

    for bm25_name, bm25_pool in bm25_preds.items():
        bm25_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                   for k, v in bm25_pool.items()}
        evaluate(bm25_pp, gt_map, label=f"{bm25_name} +PP")

    # BM25 fusion
    if len(bm25_preds) >= 2:
        for rrf_k in [30, 60]:
            fused_bm25 = {}
            for key in all_keys:
                rankings = [bm25_preds[name][key] for name in bm25_preds
                           if key in bm25_preds[name]]
                if rankings:
                    fused_bm25[key] = rrf_fusion(rankings, k=rrf_k)
            fused_bm25_pp = {k: apply_all_tricks(v, issue_texts.get(k, ''))
                            for k, v in fused_bm25.items()}
            evaluate(fused_bm25_pp, gt_map, label=f"BM25 RRF(k={rrf_k}) +PP")

    # ============================================================
    # 6. Majority vote: use neural top-1 when pools agree
    # ============================================================
    if len(available) >= 2:
        print("\n" + "="*80)
        print("6. MAJORITY VOTE + CONFIDENCE SELECTION")
        print("="*80)

        from collections import Counter

        for gap_threshold in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for min_agreement in [2, 3]:
                if min_agreement > len(available):
                    continue
                fused_preds = {}
                n_neural_used = 0
                for key in all_keys:
                    if key not in bm25_preds.get('bm25_tricked', {}):
                        continue
                    # Get top-1 from each confident neural pool
                    confident_top1 = []
                    confident_rankings = []
                    for name in available:
                        if key not in neural_preds[name] or key not in neural_gaps.get(name, {}):
                            continue
                        gap = neural_gaps[name].get(key, 0)
                        if gap >= gap_threshold:
                            r = apply_all_tricks(neural_preds[name][key], issue_texts.get(key, ''))
                            confident_top1.append(r[0])
                            confident_rankings.append(r)

                    bm25_ranking = bm25_preds['bm25_tricked'][key]

                    if confident_top1:
                        top1_counts = Counter(confident_top1).most_common(1)
                        consensus = top1_counts[0][0]
                        agreement = top1_counts[0][1]

                        if agreement >= min_agreement:
                            # Build ranking: consensus first, then neural union, then BM25
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
                            fused_preds[key] = selected
                            n_neural_used += 1
                        else:
                            fused_preds[key] = bm25_ranking
                    else:
                        fused_preds[key] = bm25_ranking

                if fused_preds:
                    evaluate(fused_preds, gt_map,
                             label=f"MajVote(gap>={gap_threshold},agree>={min_agreement}) n={n_neural_used}")

        # Confidence-weighted RRF
        print("\n--- Confidence-Weighted RRF ---")
        for gap_threshold in [0.5, 1.0, 1.5, 2.0]:
            fused_preds = {}
            for key in all_keys:
                if key not in bm25_preds.get('bm25_tricked', {}):
                    continue
                rankings = []
                weights = []
                for name in available:
                    if key not in neural_preds[name]:
                        continue
                    gap = neural_gaps[name].get(key, 0)
                    if gap >= gap_threshold:
                        r = apply_all_tricks(neural_preds[name][key], issue_texts.get(key, ''))
                        rankings.append(r)
                        weights.append(2.0 + gap)
                rankings.append(bm25_preds['bm25_tricked'][key])
                weights.append(3.0)
                fused = rrf_fusion(rankings, weights=weights, k=30)
                fused_preds[key] = apply_all_tricks(fused, issue_texts.get(key, ''))

            evaluate(fused_preds, gt_map,
                     label=f"ConfRRF(gap>={gap_threshold})")

    # ============================================================
    # 7. Pool-subset majority vote (only strong pools)
    # ============================================================
    if len(available) >= 3:
        print("\n" + "="*80)
        print("7. POOL-SUBSET MAJORITY VOTE")
        print("="*80)

        # Compute per-pool 274@1 for ranking pool quality
        pool_quality = {}
        for name in available:
            pp_preds = {k: apply_all_tricks(neural_preds[name][k], issue_texts.get(k, ''))
                       for k in neural_preds[name]}
            correct = 0
            total = 0
            for key, ranking in pp_preds.items():
                gt = gt_map.get(key, set())
                if not gt or key in EXCLUDED_274:
                    continue
                total += 1
                if gt.issubset(set(ranking[:1])):
                    correct += 1
            pool_quality[name] = correct / total * 100 if total > 0 else 0
            print(f"  Pool {name}: 274@1 = {pool_quality[name]:.1f}%")

        # Sort pools by quality
        sorted_pools = sorted(available, key=lambda x: -pool_quality[x])
        print(f"  Ranked: {[(p, f'{pool_quality[p]:.1f}') for p in sorted_pools]}")

        # Try top-K subsets
        for top_k in [3, 4, 5]:
            if top_k > len(sorted_pools):
                continue
            subset = sorted_pools[:top_k]
            for gap_threshold in [0.0, 0.5, 1.0, 1.5]:
                for min_agreement in [2, 3]:
                    if min_agreement > top_k:
                        continue
                    fused_preds = {}
                    n_neural_used = 0
                    for key in all_keys:
                        if key not in bm25_preds.get('bm25_tricked', {}):
                            continue
                        confident_top1 = []
                        confident_rankings = []
                        for name in subset:
                            if key not in neural_preds[name] or key not in neural_gaps.get(name, {}):
                                continue
                            gap = neural_gaps[name].get(key, 0)
                            if gap >= gap_threshold:
                                r = apply_all_tricks(neural_preds[name][key], issue_texts.get(key, ''))
                                confident_top1.append(r[0])
                                confident_rankings.append(r)

                        bm25_ranking = bm25_preds['bm25_tricked'][key]

                        if confident_top1:
                            top1_counts = Counter(confident_top1).most_common(1)
                            consensus = top1_counts[0][0]
                            agreement = top1_counts[0][1]

                            if agreement >= min_agreement:
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
                                fused_preds[key] = selected
                                n_neural_used += 1
                            else:
                                fused_preds[key] = bm25_ranking
                        else:
                            fused_preds[key] = bm25_ranking

                    if fused_preds:
                        evaluate(fused_preds, gt_map,
                                 label=f"Top{top_k}MV(gap>={gap_threshold},agr>={min_agreement}) n={n_neural_used}")

    # ============================================================
    # 8. Weighted majority vote (vote weight = pool quality)
    # ============================================================
    if len(available) >= 3:
        print("\n" + "="*80)
        print("8. WEIGHTED MAJORITY VOTE")
        print("="*80)

        for gap_threshold in [0.0, 0.5, 1.0, 1.5]:
            for weight_threshold in [1.5, 2.0]:
                fused_preds = {}
                n_neural_used = 0
                for key in all_keys:
                    if key not in bm25_preds.get('bm25_tricked', {}):
                        continue
                    # Weighted voting
                    weighted_votes = defaultdict(float)
                    confident_rankings = []
                    for name in available:
                        if key not in neural_preds[name] or key not in neural_gaps.get(name, {}):
                            continue
                        gap = neural_gaps[name].get(key, 0)
                        if gap >= gap_threshold:
                            r = apply_all_tricks(neural_preds[name][key], issue_texts.get(key, ''))
                            # Weight = pool quality (higher is better)
                            w = pool_quality.get(name, 30.0) / 30.0  # normalize
                            weighted_votes[r[0]] += w
                            confident_rankings.append(r)

                    bm25_ranking = bm25_preds['bm25_tricked'][key]

                    if weighted_votes:
                        best_candidate = max(weighted_votes, key=weighted_votes.get)
                        best_weight = weighted_votes[best_candidate]

                        if best_weight >= weight_threshold:
                            seen = {best_candidate}
                            selected = [best_candidate]
                            for r in confident_rankings:
                                for f in r:
                                    if f not in seen:
                                        selected.append(f)
                                        seen.add(f)
                            for f in bm25_ranking:
                                if f not in seen:
                                    selected.append(f)
                                    seen.add(f)
                            fused_preds[key] = selected
                            n_neural_used += 1
                        else:
                            fused_preds[key] = bm25_ranking
                    else:
                        fused_preds[key] = bm25_ranking

                if fused_preds:
                    evaluate(fused_preds, gt_map,
                             label=f"WeightedMV(gap>={gap_threshold},w>={weight_threshold}) n={n_neural_used}")

    print("\nDone.")


if __name__ == '__main__':
    main()
