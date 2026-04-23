#!/usr/bin/env python3
"""
Compile and compare all experiment results.

Checks all experiment directories for summary.json and prediction files,
computes metrics, and produces a comprehensive comparison table.
"""
import os
import json
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LocAgent excluded instances for matched protocol
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


def compute_metrics_from_predictions(pred_path, k_values=[1, 3, 5, 10, 20]):
    """Compute metrics from predictions.jsonl file."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))

    metrics = {}
    for k in k_values:
        recalls = []
        for p in preds:
            gt = set(p.get('ground_truth', p.get('gt_files', [])))
            pred = p.get('predicted', p.get('reranked', []))[:k]
            if gt:
                recalls.append(len(set(pred) & gt) / len(gt))
        metrics[f'recall@{k}'] = np.mean(recalls) * 100 if recalls else 0

    # Cond acc@1
    correct, total = 0, 0
    for p in preds:
        gt = set(p.get('ground_truth', p.get('gt_files', [])))
        pred = p.get('predicted', p.get('reranked', []))
        candidates = p.get('candidates', pred)
        if gt & set(candidates):
            total += 1
            if pred and pred[0] in gt:
                correct += 1
    metrics['cond_acc@1'] = correct / max(total, 1) * 100
    metrics['n_examples'] = len(preds)

    return metrics


def compile_swebench_results():
    """Compile all SWE-bench related results."""
    print("=" * 80)
    print("SWE-BENCH LITE RESULTS")
    print("=" * 80)

    # BM25 results
    bm25_files = {
        'Path-only BM25': 'data/rankft/swebench_test_bm25_top500.jsonl',
        'Content BM25': 'data/rankft/swebench_bm25_content_top500.jsonl',
        'Function BM25': 'data/rankft/swebench_bm25_function_top500.jsonl',
        'Function BM25 (no test)': 'data/rankft/swebench_bm25_function_notest_top500.jsonl',
        'Chunk BM25 (no test)': 'data/rankft/swebench_bm25_chunk_top500.jsonl',
        'Best Ensemble': 'data/rankft/swebench_bm25_best_ensemble_top500.jsonl',
        'Best + Tricks': 'data/rankft/swebench_bm25_final_top500.jsonl',
    }

    print("\n--- BM25 Retrieval (before reranking) ---")
    print(f"{'Method':<30} {'@1':>7} {'@5':>7} {'@10':>7} {'@20':>7} {'@50':>7}  |  {'274@1':>7} {'274@5':>7}")
    print("-" * 100)

    for name, path in bm25_files.items():
        full_path = os.path.join(BASE_DIR, path)
        if not os.path.exists(full_path):
            continue
        data = [json.loads(l) for l in open(full_path)]
        matched = [d for d in data if d['issue_id'] not in EXCLUDED_274]

        def acc_at_k(items, k):
            return sum(1 for d in items if set(d['ground_truth']).issubset(set(d['bm25_candidates'][:k]))) / len(items) * 100

        all_accs = {k: acc_at_k(data, k) for k in [1, 5, 10, 20, 50]}
        m_accs = {k: acc_at_k(matched, k) for k in [1, 5]}

        print(f"{name:<30} {all_accs[1]:>7.2f} {all_accs[5]:>7.2f} {all_accs[10]:>7.2f} "
              f"{all_accs[20]:>7.2f} {all_accs[50]:>7.2f}  |  {m_accs[1]:>7.2f} {m_accs[5]:>7.2f}")

    # Neural reranking results
    rerank_dirs = {
        'Rerank (path BM25)': 'experiments/rankft_runB_graph/eval_swebench_500',
        'Rerank (content BM25)': 'experiments/rankft_runB_graph/eval_swebench_content_bm25',
        'Rerank (function BM25)': 'experiments/rankft_runB_graph/eval_swebench_function_bm25',
        'Rerank (func notest)': 'experiments/rankft_runB_graph/eval_swebench_funcnotest',
        'Rerank (chunk BM25)': 'experiments/rankft_runB_graph/eval_swebench_chunk',
        'Rerank (best ensemble)': 'experiments/rankft_runB_graph/eval_swebench_best_ensemble',
    }

    print(f"\n--- Neural Reranking ---")
    print(f"{'Method':<30} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'C.Acc@1':>8}  |  {'274@1':>7}")
    print("-" * 100)

    for name, dir_path in rerank_dirs.items():
        full_dir = os.path.join(BASE_DIR, dir_path)
        summary_path = os.path.join(full_dir, 'summary.json')
        pred_path = os.path.join(full_dir, 'predictions.jsonl')

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            overall = s.get('overall', s)
            r1 = overall.get('recall@1', overall.get('hit@1', 0))
            r5 = overall.get('recall@5', overall.get('hit@5', 0))
            r10 = overall.get('recall@10', overall.get('hit@10', 0))
            r20 = overall.get('recall@20', overall.get('hit@20', 0))
            ca = overall.get('cond_acc@1|gt_in_candidates', overall.get('cond_acc@1', 0))

            # Compute matched 274 if predictions exist
            m_r1 = "?"
            if os.path.exists(pred_path):
                preds = [json.loads(l) for l in open(pred_path)]
                matched = [p for p in preds
                          if p.get('issue_id', p.get('instance_id', '')) not in EXCLUDED_274]
                if matched:
                    recalls = []
                    for p in matched:
                        gt = set(p.get('ground_truth', []))
                        pred_files = p.get('predicted', p.get('reranked', []))[:1]
                        if gt:
                            recalls.append(len(set(pred_files) & gt) / len(gt))
                    m_r1 = f"{np.mean(recalls)*100:.2f}" if recalls else "?"

            print(f"{name:<30} {r1:>7.2f} {r5:>7.2f} {r10:>7.2f} {r20:>7.2f} {ca:>8.2f}  |  {m_r1:>7}")
        elif os.path.exists(pred_path):
            m = compute_metrics_from_predictions(pred_path)
            print(f"{name:<30} {m['recall@1']:>7.2f} {m['recall@5']:>7.2f} {m['recall@10']:>7.2f} "
                  f"{m['recall@20']:>7.2f} {m['cond_acc@1']:>8.2f}  |  {'?':>7}")
        else:
            print(f"{name:<30} {'(running)':>50}")

    # LocAgent comparison
    print(f"\n--- LocAgent Comparison (matched 274) ---")
    print(f"{'Method':<35} {'Acc@1':>7} {'Acc@5':>7}")
    print("-" * 50)
    print(f"{'LocAgent BM25':<35} {'38.69':>7} {'61.68':>7}")
    print(f"{'CodeRankEmbed':<35} {'52.55':>7} {'':>7}")
    print(f"{'LocAgent (Qwen2.5-7B)':<35} {'70.80':>7} {'':>7}")
    print(f"{'LocAgent (Claude-3.5)':<35} {'77.37':>7} {'':>7}")


def compile_grepo_results():
    """Compile all GREPO results."""
    print("\n\n" + "=" * 80)
    print("GREPO RESULTS")
    print("=" * 80)

    result_dirs = {
        'Baseline (merged candidates)': 'experiments/rankft_runB_graph/eval_merged_rerank',
        'Baseline (exp6 candidates)': 'experiments/rankft_runB_graph/eval_exp6_rerank',
        'BM25-500 candidates': 'experiments/rankft_runB_graph/eval_bm25_500_rerank',
        'Closed-book': 'experiments/rankft_runB_graph/eval_closedbook',
    }

    print(f"\n--- Standard Reranking ---")
    print(f"{'Method':<35} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'C.Acc@1':>8}")
    print("-" * 75)

    for name, dir_path in result_dirs.items():
        full_dir = os.path.join(BASE_DIR, dir_path)
        summary_path = os.path.join(full_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            o = s.get('overall', s)
            r1 = o.get('recall@1', o.get('hit@1', 0))
            r5 = o.get('recall@5', o.get('hit@5', 0))
            r10 = o.get('recall@10', o.get('hit@10', 0))
            r20 = o.get('recall@20', o.get('hit@20', 0))
            ca = o.get('cond_acc@1|gt_in_candidates', o.get('cond_acc@1', 0))
            print(f"{name:<35} {r1:>7.2f} {r5:>7.2f} {r10:>7.2f} {r20:>7.2f} {ca:>8.2f}")
        else:
            print(f"{name:<35} {'(running)':>40}")

    # Score combination
    sc_dir = os.path.join(BASE_DIR, 'experiments/score_combination')
    if os.path.exists(os.path.join(sc_dir, 'summary.json')):
        print(f"\n--- Score Combination ---")
        with open(os.path.join(sc_dir, 'summary.json')) as f:
            s = json.load(f)
        for strategy, metrics in s.items():
            if isinstance(metrics, dict) and 'recall@1' in metrics:
                print(f"  {strategy:<30} R@1={metrics['recall@1']:.2f}%")

    # Two-stage
    ts_dir = os.path.join(BASE_DIR, 'experiments/twostage_rerank')
    if os.path.exists(os.path.join(ts_dir, 'summary.json')):
        print(f"\n--- Two-Stage Reranking ---")
        with open(os.path.join(ts_dir, 'summary.json')) as f:
            s = json.load(f)
        for strategy, metrics in s.items():
            if isinstance(metrics, dict) and 'recall@1' in metrics:
                r1 = metrics.get('recall@1', 0)
                r5 = metrics.get('recall@5', 0)
                print(f"  {strategy:<30} R@1={r1:.2f}% R@5={r5:.2f}%")

    # Fair comparison
    fair_dir = os.path.join(BASE_DIR, 'experiments/graph_rag_fair_full_4cond')
    if os.path.exists(os.path.join(fair_dir, 'summary.json')):
        print(f"\n--- Fair Comparison (4 conditions) ---")
        with open(os.path.join(fair_dir, 'summary.json')) as f:
            s = json.load(f)
        for cond, metrics in s.items():
            if isinstance(metrics, dict) and 'recall@1' in metrics:
                r1 = metrics.get('recall@1', 0)
                print(f"  {cond:<30} R@1={r1:.2f}%")


def compile_ablation_results():
    """Compile ablation and analysis results."""
    print("\n\n" + "=" * 80)
    print("ABLATION / ANALYSIS RESULTS")
    print("=" * 80)

    # Path anonymization
    print(f"\n--- Path Anonymization Control ---")
    for cond in ['original', 'hashed', 'shuffled']:
        path = os.path.join(BASE_DIR, f'experiments/path_anonymized/{cond}/summary.json')
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            r1 = s.get('overall', s).get('recall@1', 0)
            ca = s.get('cond_acc1', 0)
            n = s.get('n_samples', 0)
            print(f"  {cond:<15} R@1={r1:.2f}%  C.Acc@1={ca:.2f}%  (n={n})")
        else:
            print(f"  {cond:<15} (running)")


def main():
    compile_swebench_results()
    compile_grepo_results()
    compile_ablation_results()
    print("\n" + "=" * 80)
    print("COMPILATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
