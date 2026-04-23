"""
Graph Feature Late Fusion: a structural scorer-side baseline for CodeGRIP.

Instead of dumping graph edges into the prompt (strawman), this computes
explicit graph features for each candidate file and trains a logistic
regression to combine cross-encoder scores with structural features.

Features per candidate file f, given cross-encoder top-k predictions T:
  1. ce_score:       original cross-encoder score (logit_yes - logit_no)
  2. ce_rank:        1/(rank+1) of the candidate in the CE ranking
  3. cc_min_dist:    1/(shortest co-change distance to any file in T)
  4. cc_neighbors:   # co-change neighbors of f that are in the candidate pool
  5. cc_weight_sum:  sum of co-change frequencies between f and files in T
  6. imp_neighbors:  # import connections between f and files in T
  7. same_dir:       max(1/dir_size) over T files sharing a directory with f
  8. same_dir_count: # files in T sharing a directory with f

Evaluation: leave-one-repo-out cross-validation (no data leakage).

Usage:
  python scripts/graph_feature_fusion.py \
    --predictions experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl \
    --train_data data/grepo_text/grepo_train.jsonl \
    --dep_graph_dir data/dep_graphs \
    --file_tree_dir data/file_trees \
    --top_k 10
"""

import json
import argparse
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Deterministic
np.random.seed(42)

# Suppress convergence warnings during CV
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Graph index builders (reused from existing codebase)
# ============================================================

def build_cochange_index(train_data_path: str, min_cochange: int = 1) -> Dict:
    """Build per-repo co-change index: repo -> file -> {neighbor: score}."""
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


def build_cochange_raw_counts(train_data_path: str) -> Dict:
    """Build per-repo raw co-change counts: repo -> (fa, fb) -> count."""
    repo_cochanges = defaultdict(lambda: defaultdict(int))

    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get('split') != 'train':
                continue
            repo = item['repo']
            files = item.get('changed_py_files', [])
            if not files:
                files = [ff for ff in item.get('changed_files', []) if ff.endswith('.py')]

            for i, fa in enumerate(files):
                for j, fb in enumerate(files):
                    if i != j:
                        repo_cochanges[repo][(fa, fb)] += 1

    return dict(repo_cochanges)


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


def build_dir_index(file_tree_dir: str) -> Tuple[Dict, Dict]:
    """Build per-repo directory->files index and all py files set."""
    dir_idx = {}
    all_py = {}

    if not os.path.isdir(file_tree_dir):
        return dir_idx, all_py

    for fname in os.listdir(file_tree_dir):
        if not fname.endswith('.json'):
            continue
        repo = fname.replace('.json', '')

        with open(os.path.join(file_tree_dir, fname)) as f:
            ft = json.load(f)

        dir_files = defaultdict(list)
        py_set = set(ft.get('py_files', []))
        all_py[repo] = py_set
        for py_file in py_set:
            d = os.path.dirname(py_file)
            dir_files[d].append(py_file)

        dir_idx[repo] = dict(dir_files)

    return dir_idx, all_py


# ============================================================
# Feature extraction
# ============================================================

def compute_features_for_example(
    pred: dict,
    cc_index: Dict,
    cc_raw: Dict,
    imp_index: Dict,
    dir_index: Dict,
    all_py: Dict,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature matrix X (n_candidates x n_features) and label vector y
    for one prediction example.

    Returns:
        X: (n_candidates, 8) feature matrix
        y: (n_candidates,) binary labels (1 if ground truth)
    """
    repo = pred['repo']
    files = pred['predicted']  # top-50, already ranked by CE
    scores = pred['scores']
    gt = set(pred['ground_truth'])

    n = len(files)
    top_k_files = set(files[:top_k])

    # Graph data for this repo
    repo_cc = cc_index.get(repo, {})
    repo_cc_raw = cc_raw.get(repo, {})
    repo_imp = imp_index.get(repo, {})
    repo_dir = dir_index.get(repo, {})

    # Pre-compute directory membership for top-k
    topk_dirs = {}  # dir -> list of top-k files in that dir
    for tf in top_k_files:
        d = os.path.dirname(tf)
        topk_dirs.setdefault(d, []).append(tf)

    # Compute directory sizes
    dir_sizes = {}
    for d in topk_dirs:
        dir_sizes[d] = len(repo_dir.get(d, [d]))  # fallback to 1

    X = np.zeros((n, 8), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    candidate_set = set(files)

    for i, f in enumerate(files):
        # Feature 0: CE score (raw)
        X[i, 0] = scores[i]

        # Feature 1: CE rank feature = 1/(rank+1)
        X[i, 1] = 1.0 / (i + 1)

        # Feature 2: shortest co-change distance to top-k
        # Direct co-change = distance 1, 2-hop = distance 2
        # We compute: max(cc_score to any top-k file)  [proxy for min distance]
        max_cc_to_topk = 0.0
        for tf in top_k_files:
            cc_score = repo_cc.get(f, {}).get(tf, 0.0)
            max_cc_to_topk = max(max_cc_to_topk, cc_score)
        X[i, 2] = max_cc_to_topk

        # Feature 3: # co-change neighbors in the candidate pool
        cc_neighbors_in_pool = 0
        for neighbor in repo_cc.get(f, {}):
            if neighbor in candidate_set and neighbor != f:
                cc_neighbors_in_pool += 1
        X[i, 3] = cc_neighbors_in_pool

        # Feature 4: co-change weight sum with top-k
        cc_weight_sum = 0.0
        for tf in top_k_files:
            raw_count = repo_cc_raw.get((f, tf), 0)
            cc_weight_sum += raw_count
        X[i, 4] = cc_weight_sum

        # Feature 5: # import connections to top-k
        imp_to_topk = 0
        imp_neighbors = repo_imp.get(f, set())
        for tf in top_k_files:
            if tf in imp_neighbors:
                imp_to_topk += 1
        X[i, 5] = imp_to_topk

        # Feature 6: same-directory indicator with top-k (best specificity)
        f_dir = os.path.dirname(f)
        if f_dir in topk_dirs:
            dir_size = max(dir_sizes.get(f_dir, 1), 1)
            X[i, 6] = 1.0 / dir_size
        else:
            X[i, 6] = 0.0

        # Feature 7: # top-k files sharing directory
        X[i, 7] = len(topk_dirs.get(f_dir, []))

        # Label
        y[i] = 1.0 if f in gt else 0.0

    return X, y


# ============================================================
# Metrics
# ============================================================

def recall_at_k(ranked_files: List[str], gt: set, k: int) -> float:
    """Fraction of GT files found in top-k."""
    if not gt:
        return 0.0
    topk = set(ranked_files[:k])
    return len(topk & gt) / len(gt)


def compute_metrics(predictions: List[dict], key='predicted') -> Dict[str, float]:
    """Compute average R@1, R@5, R@10 across predictions."""
    metrics = {f'R@{k}': [] for k in [1, 5, 10]}
    for p in predictions:
        gt = set(p['ground_truth'])
        ranked = p[key]
        for k in [1, 5, 10]:
            metrics[f'R@{k}'].append(recall_at_k(ranked, gt, k))

    return {k: np.mean(v) * 100 for k, v in metrics.items()}


# ============================================================
# Fusion model: leave-one-repo-out
# ============================================================

def run_fusion_loro(
    predictions: List[dict],
    cc_index: Dict,
    cc_raw: Dict,
    imp_index: Dict,
    dir_index: Dict,
    all_py: Dict,
    top_k: int = 10,
    C: float = 1.0,
) -> Tuple[Dict, List[dict]]:
    """
    Leave-one-repo-out cross-validation.
    Train LR on all repos except one, predict on held-out repo.
    Re-rank by predicted probability.
    """
    # Group predictions by repo
    repo_preds = defaultdict(list)
    for p in predictions:
        repo_preds[p['repo']].append(p)

    repos = sorted(repo_preds.keys())
    print(f"  LORO CV over {len(repos)} repos")

    # Pre-compute features for all examples
    all_features = {}  # (repo, issue_id) -> (X, y, pred)
    for p in predictions:
        key = (p['repo'], p['issue_id'])
        X, y_vec = compute_features_for_example(
            p, cc_index, cc_raw, imp_index, dir_index, all_py, top_k
        )
        all_features[key] = (X, y_vec, p)

    # LORO
    fused_predictions = []
    feature_importances = []

    for held_out_repo in repos:
        # Collect train data (all repos except held-out)
        X_train_parts = []
        y_train_parts = []
        for repo_name in repos:
            if repo_name == held_out_repo:
                continue
            for p in repo_preds[repo_name]:
                key = (p['repo'], p['issue_id'])
                X, y_vec, _ = all_features[key]
                X_train_parts.append(X)
                y_train_parts.append(y_vec)

        if not X_train_parts:
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Check we have both classes
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            # Degenerate: just use CE scores for this repo
            for p in repo_preds[held_out_repo]:
                fused_predictions.append(dict(p))
            continue

        # Fit scaler + LR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(
            C=C, max_iter=1000, solver='lbfgs', random_state=42,
            class_weight='balanced',
        )
        clf.fit(X_train_scaled, y_train)
        feature_importances.append(clf.coef_[0].copy())

        # Predict on held-out repo
        for p in repo_preds[held_out_repo]:
            key = (p['repo'], p['issue_id'])
            X_test, _, _ = all_features[key]
            X_test_scaled = scaler.transform(X_test)
            proba = clf.predict_proba(X_test_scaled)[:, 1]

            # Re-rank by fusion probability
            sorted_idx = np.argsort(-proba)
            new_ranked = [p['predicted'][i] for i in sorted_idx]

            new_p = dict(p)
            new_p['predicted_fused'] = new_ranked
            new_p['fusion_scores'] = proba[sorted_idx].tolist()
            fused_predictions.append(new_p)

    # Aggregate feature importances
    if feature_importances:
        avg_coef = np.mean(feature_importances, axis=0)
        std_coef = np.std(feature_importances, axis=0)
    else:
        avg_coef = np.zeros(8)
        std_coef = np.zeros(8)

    feature_names = [
        'ce_score', 'ce_rank', 'cc_max_to_topk', 'cc_neighbors_pool',
        'cc_weight_sum', 'imp_neighbors_topk', 'same_dir_specificity',
        'same_dir_count',
    ]

    coef_info = {name: (avg_coef[i], std_coef[i]) for i, name in enumerate(feature_names)}

    # Compute metrics
    baseline_metrics = compute_metrics(fused_predictions, key='predicted')
    fused_metrics = compute_metrics(fused_predictions, key='predicted_fused')

    return {
        'baseline': baseline_metrics,
        'fused': fused_metrics,
        'coefficients': coef_info,
        'n_examples': len(fused_predictions),
    }, fused_predictions


# ============================================================
# Fixed-formula fusion (no learning, for sanity check)
# ============================================================

def run_fixed_formula_fusion(
    predictions: List[dict],
    cc_index: Dict,
    cc_raw: Dict,
    imp_index: Dict,
    dir_index: Dict,
    all_py: Dict,
    top_k: int = 10,
    alpha: float = 0.8,
) -> Dict:
    """
    Fixed formula: fused_score = alpha * norm(ce_score) + (1-alpha) * norm(graph_score)
    where graph_score = sum of features 2-7 (normalized).
    No training -- a pure heuristic baseline.
    """
    results = []
    for p in predictions:
        X, _ = compute_features_for_example(
            p, cc_index, cc_raw, imp_index, dir_index, all_py, top_k
        )

        # Normalize CE score to [0,1]
        ce = X[:, 0]
        ce_range = ce.max() - ce.min()
        if ce_range > 0:
            ce_norm = (ce - ce.min()) / ce_range
        else:
            ce_norm = np.ones_like(ce)

        # Graph features: sum of features 2-7, each normalized
        graph_score = np.zeros(len(ce))
        for feat_idx in range(2, 8):
            feat = X[:, feat_idx]
            feat_range = feat.max() - feat.min()
            if feat_range > 0:
                graph_score += (feat - feat.min()) / feat_range
            # If constant, contributes 0 (no signal)

        # Normalize graph_score
        gs_range = graph_score.max() - graph_score.min()
        if gs_range > 0:
            graph_norm = (graph_score - graph_score.min()) / gs_range
        else:
            graph_norm = np.zeros_like(graph_score)

        fused = alpha * ce_norm + (1 - alpha) * graph_norm

        sorted_idx = np.argsort(-fused)
        new_ranked = [p['predicted'][i] for i in sorted_idx]

        new_p = dict(p)
        new_p['predicted_fused'] = new_ranked
        results.append(new_p)

    baseline_metrics = compute_metrics(results, key='predicted')
    fused_metrics = compute_metrics(results, key='predicted_fused')

    return {
        'baseline': baseline_metrics,
        'fused': fused_metrics,
        'n_examples': len(results),
    }


# ============================================================
# Per-repo breakdown
# ============================================================

def per_repo_breakdown(fused_predictions: List[dict]) -> Dict:
    """Compute per-repo metrics for baseline and fused."""
    repo_preds = defaultdict(list)
    for p in fused_predictions:
        repo_preds[p['repo']].append(p)

    breakdown = {}
    for repo in sorted(repo_preds.keys()):
        preds = repo_preds[repo]
        base = compute_metrics(preds, key='predicted')
        fused = compute_metrics(preds, key='predicted_fused')
        breakdown[repo] = {
            'count': len(preds),
            'baseline_R@1': base['R@1'],
            'fused_R@1': fused['R@1'],
            'delta_R@1': fused['R@1'] - base['R@1'],
        }

    return breakdown


# ============================================================
# Feature ablation
# ============================================================

def run_ablation(
    predictions: List[dict],
    cc_index: Dict,
    cc_raw: Dict,
    imp_index: Dict,
    dir_index: Dict,
    all_py: Dict,
    top_k: int = 10,
    C: float = 1.0,
) -> Dict:
    """
    Ablation: train with subsets of features.
    - CE only (features 0,1)
    - CE + co-change (features 0,1,2,3,4)
    - CE + import (features 0,1,5)
    - CE + directory (features 0,1,6,7)
    - All features
    """
    feature_groups = {
        'CE only':          [0, 1],
        'CE + co-change':   [0, 1, 2, 3, 4],
        'CE + import':      [0, 1, 5],
        'CE + directory':   [0, 1, 6, 7],
        'All features':     list(range(8)),
    }

    # Pre-compute features
    repo_preds = defaultdict(list)
    for p in predictions:
        repo_preds[p['repo']].append(p)
    repos = sorted(repo_preds.keys())

    all_features = {}
    for p in predictions:
        key = (p['repo'], p['issue_id'])
        X, y_vec = compute_features_for_example(
            p, cc_index, cc_raw, imp_index, dir_index, all_py, top_k
        )
        all_features[key] = (X, y_vec, p)

    ablation_results = {}

    for group_name, feat_indices in feature_groups.items():
        fused_predictions = []

        for held_out_repo in repos:
            X_train_parts = []
            y_train_parts = []
            for repo_name in repos:
                if repo_name == held_out_repo:
                    continue
                for p in repo_preds[repo_name]:
                    key = (p['repo'], p['issue_id'])
                    X, y_vec, _ = all_features[key]
                    X_train_parts.append(X[:, feat_indices])
                    y_train_parts.append(y_vec)

            if not X_train_parts:
                continue

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                for p in repo_preds[held_out_repo]:
                    new_p = dict(p)
                    new_p['predicted_fused'] = list(p['predicted'])
                    fused_predictions.append(new_p)
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            clf = LogisticRegression(
                C=C, max_iter=1000, solver='lbfgs', random_state=42,
                class_weight='balanced',
            )
            clf.fit(X_train_scaled, y_train)

            for p in repo_preds[held_out_repo]:
                key = (p['repo'], p['issue_id'])
                X_test, _, _ = all_features[key]
                X_test_scaled = scaler.transform(X_test[:, feat_indices])
                proba = clf.predict_proba(X_test_scaled)[:, 1]

                sorted_idx = np.argsort(-proba)
                new_ranked = [p['predicted'][i] for i in sorted_idx]

                new_p = dict(p)
                new_p['predicted_fused'] = new_ranked
                fused_predictions.append(new_p)

        fused_metrics = compute_metrics(fused_predictions, key='predicted_fused')
        ablation_results[group_name] = fused_metrics

    return ablation_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Graph Feature Late Fusion: scorer-side structural baseline'
    )
    parser.add_argument('--predictions', required=True,
                        help='Path to predictions.jsonl from cross-encoder eval')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl',
                        help='Training data for co-change graph')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs',
                        help='Directory with import graph files')
    parser.add_argument('--file_tree_dir', default='data/file_trees',
                        help='Directory with file tree JSONs')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-k CE predictions to use as anchors for graph features')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularization strength')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save fused predictions (optional)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run feature ablation study')
    parser.add_argument('--sweep_alpha', action='store_true',
                        help='Sweep fixed-formula alpha values')
    parser.add_argument('--sweep_topk', action='store_true',
                        help='Sweep top-k anchor values')
    args = parser.parse_args()

    # ---- Load predictions ----
    print("Loading predictions...")
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            p = json.loads(line)
            # Only keep examples with scores (reranked)
            if p.get('scores'):
                predictions.append(p)
    print(f"  {len(predictions)} examples loaded")

    # ---- Build graph indices ----
    print("Building co-change index...")
    cc_index = build_cochange_index(args.train_data)
    print(f"  {len(cc_index)} repos with co-change data")

    print("Building co-change raw counts...")
    cc_raw = build_cochange_raw_counts(args.train_data)
    print(f"  {len(cc_raw)} repos")

    print("Building import index...")
    imp_index = build_import_index(args.dep_graph_dir)
    print(f"  {len(imp_index)} repos with import data")

    print("Building directory index...")
    dir_index, all_py = build_dir_index(args.file_tree_dir)
    print(f"  {len(dir_index)} repos with file tree data")

    # ---- Baseline metrics (CE only) ----
    baseline = compute_metrics(predictions, key='predicted')
    print(f"\n{'='*70}")
    print("CROSS-ENCODER BASELINE (no fusion):")
    print(f"  R@1={baseline['R@1']:.2f}%  R@5={baseline['R@5']:.2f}%  R@10={baseline['R@10']:.2f}%")

    # ---- Feature statistics ----
    print(f"\nFeature coverage analysis:")
    has_cc = sum(1 for p in predictions if p['repo'] in cc_index)
    has_imp = sum(1 for p in predictions if p['repo'] in imp_index)
    has_dir = sum(1 for p in predictions if p['repo'] in dir_index)
    print(f"  Examples with co-change data: {has_cc}/{len(predictions)}")
    print(f"  Examples with import data:    {has_imp}/{len(predictions)}")
    print(f"  Examples with file tree data: {has_dir}/{len(predictions)}")

    # ---- LORO Logistic Regression Fusion ----
    print(f"\n{'='*70}")
    print(f"LOGISTIC REGRESSION FUSION (LORO CV, top_k={args.top_k}, C={args.C}):")
    print(f"{'='*70}")

    lr_results, fused_preds = run_fusion_loro(
        predictions, cc_index, cc_raw, imp_index, dir_index, all_py,
        top_k=args.top_k, C=args.C,
    )

    base = lr_results['baseline']
    fused = lr_results['fused']
    print(f"\n  {'Metric':<8} {'CE-only':>10} {'CE+Graph':>10} {'Delta':>10}")
    print(f"  {'-'*40}")
    for k in [1, 5, 10]:
        key = f'R@{k}'
        delta = fused[key] - base[key]
        print(f"  {key:<8} {base[key]:>9.2f}% {fused[key]:>9.2f}% {delta:>+9.2f}%")

    print(f"\n  Feature coefficients (mean +/- std across LORO folds):")
    for name, (mean, std) in lr_results['coefficients'].items():
        print(f"    {name:<25s} {mean:>+8.4f} +/- {std:.4f}")

    # ---- Per-repo breakdown ----
    print(f"\n  Per-repo R@1 changes (top movers):")
    breakdown = per_repo_breakdown(fused_preds)
    sorted_repos = sorted(breakdown.items(), key=lambda x: abs(x[1]['delta_R@1']), reverse=True)
    print(f"  {'Repo':<25s} {'Count':>5} {'Base':>7} {'Fused':>7} {'Delta':>7}")
    print(f"  {'-'*55}")
    for repo, info in sorted_repos[:15]:
        print(f"  {repo:<25s} {info['count']:>5d} {info['baseline_R@1']:>6.1f}% "
              f"{info['fused_R@1']:>6.1f}% {info['delta_R@1']:>+6.1f}%")

    # Count improved/degraded repos
    improved = sum(1 for _, info in breakdown.items() if info['delta_R@1'] > 0.5)
    degraded = sum(1 for _, info in breakdown.items() if info['delta_R@1'] < -0.5)
    unchanged = len(breakdown) - improved - degraded
    print(f"\n  Repos improved: {improved}, degraded: {degraded}, unchanged: {unchanged}")

    # ---- Fixed formula sweep ----
    if args.sweep_alpha:
        print(f"\n{'='*70}")
        print("FIXED FORMULA SWEEP (alpha * CE + (1-alpha) * graph):")
        print(f"{'='*70}")
        print(f"  {'alpha':>6} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'dR@1':>8}")
        print(f"  {'-'*42}")
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            ff_result = run_fixed_formula_fusion(
                predictions, cc_index, cc_raw, imp_index, dir_index, all_py,
                top_k=args.top_k, alpha=alpha,
            )
            f_m = ff_result['fused']
            delta = f_m['R@1'] - base['R@1']
            print(f"  {alpha:>6.2f} {f_m['R@1']:>7.2f}% {f_m['R@5']:>7.2f}% "
                  f"{f_m['R@10']:>7.2f}% {delta:>+7.2f}%")

    # ---- Top-k anchor sweep ----
    if args.sweep_topk:
        print(f"\n{'='*70}")
        print("TOP-K ANCHOR SWEEP (LR fusion):")
        print(f"{'='*70}")
        print(f"  {'top_k':>6} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'dR@1':>8}")
        print(f"  {'-'*42}")
        for tk in [3, 5, 10, 15, 20]:
            tk_result, _ = run_fusion_loro(
                predictions, cc_index, cc_raw, imp_index, dir_index, all_py,
                top_k=tk, C=args.C,
            )
            f_m = tk_result['fused']
            delta = f_m['R@1'] - base['R@1']
            print(f"  {tk:>6d} {f_m['R@1']:>7.2f}% {f_m['R@5']:>7.2f}% "
                  f"{f_m['R@10']:>7.2f}% {delta:>+7.2f}%")

    # ---- Feature ablation ----
    if args.ablation:
        print(f"\n{'='*70}")
        print("FEATURE ABLATION (LORO LR):")
        print(f"{'='*70}")
        ablation = run_ablation(
            predictions, cc_index, cc_raw, imp_index, dir_index, all_py,
            top_k=args.top_k, C=args.C,
        )
        print(f"  {'Feature group':<25s} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
        print(f"  {'-'*50}")
        for group_name, metrics in ablation.items():
            print(f"  {group_name:<25s} {metrics['R@1']:>7.2f}% "
                  f"{metrics['R@5']:>7.2f}% {metrics['R@10']:>7.2f}%")

    # ---- Save output ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'predictions_fused.jsonl')
        with open(out_path, 'w') as f:
            for p in fused_preds:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        print(f"\nFused predictions saved to {out_path}")

        summary = {
            'baseline': base,
            'fused': fused,
            'coefficients': {
                name: {'mean': float(mean), 'std': float(std)}
                for name, (mean, std) in lr_results['coefficients'].items()
            },
            'config': {
                'top_k': args.top_k,
                'C': args.C,
                'n_examples': lr_results['n_examples'],
                'method': 'logistic_regression_LORO',
            },
        }
        summary_path = os.path.join(args.output_dir, 'fusion_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")

    print(f"\n{'='*70}")
    print("SUMMARY FOR PAPER:")
    print(f"  Cross-encoder only:      R@1={base['R@1']:.2f}%  R@5={base['R@5']:.2f}%  R@10={base['R@10']:.2f}%")
    print(f"  CE + Graph Feature LR:   R@1={fused['R@1']:.2f}%  R@5={fused['R@5']:.2f}%  R@10={fused['R@10']:.2f}%")
    delta1 = fused['R@1'] - base['R@1']
    delta5 = fused['R@5'] - base['R@5']
    delta10 = fused['R@10'] - base['R@10']
    print(f"  Delta:                   R@1={delta1:+.2f}%  R@5={delta5:+.2f}%  R@10={delta10:+.2f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
