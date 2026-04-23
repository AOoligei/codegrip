"""
Learned Reranking Pipeline for bug localization predictions.

Instead of rule-based text matching, trains a lightweight classifier
(LightGBM / Logistic Regression) on features extracted from
(issue, candidate_file) pairs to predict whether each candidate is a GT file.

Oracle ceiling: H@5=47.53% vs actual H@5=33.24% — 14.3 point gap to close.

IMPORTANT: The `generate_data` / `train` / `rerank` pipeline trains on test
labels, causing data leakage. For clean evaluation, use `cv_rerank` which
performs strict 5-fold cross-validation — each example is scored by a model
that NEVER saw its label during training.

Clean pipeline (recommended):
    python src/eval/learned_reranker.py cv_rerank \
        --predictions experiments/exp1_sft_only/eval_unified_expansion/predictions.jsonl \
        --base_predictions experiments/exp1_sft_only/eval_filetree/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl \
        --train_data data/grepo_text/grepo_train.jsonl \
        --output experiments/exp1_sft_only/eval_learned_rerank_cv/predictions.jsonl \
        --n_folds 5

Leaked pipeline (deprecated, for oracle-style upper-bound only):
    python src/eval/learned_reranker.py generate_data ...
    python src/eval/learned_reranker.py train ...
    python src/eval/learned_reranker.py rerank ...
"""

import json
import os
import re
import argparse
import pickle
import random
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import numpy as np

random.seed(42)
np.random.seed(42)


# ========== Feature Extraction ==========

def extract_file_mentions(issue_text: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract file paths, partial names, and keywords from issue text."""
    full_paths = set()
    partial_names = set()
    keywords = set()

    lines = issue_text.split('\n')
    title = lines[0] if lines else ''

    for m in re.finditer(r'[\w./\\-]+\.py\b', issue_text):
        path = m.group(0).lstrip('./')
        if '/' in path:
            full_paths.add(path)
        partial_names.add(os.path.basename(path))

    for m in re.finditer(r'`([\w.]+)`', issue_text):
        name = m.group(1)
        parts = name.split('.')
        if len(parts) >= 2:
            full_paths.add('/'.join(parts) + '.py')
            for part in parts:
                if len(part) > 2:
                    keywords.add(part.lower())
        elif len(name) > 2:
            keywords.add(name.lower())

    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', title):
        keywords.add(m.group(1).lower())
    for m in re.finditer(r'\b([a-z]\w*_\w+)\b', title):
        keywords.add(m.group(1).lower())

    return full_paths, partial_names, keywords


def compute_text_score(filepath: str, full_paths: Set[str],
                       partial_names: Set[str], keywords: Set[str]) -> float:
    """Compute text relevance score for a file."""
    score = 0.0
    basename = os.path.basename(filepath)
    stem = basename.replace('.py', '').lower()
    path_parts = filepath.lower().replace('.py', '').split('/')
    path_tokens = set()
    for p in path_parts:
        path_tokens.add(p)
        for t in p.split('_'):
            if t:
                path_tokens.add(t)

    for fp in full_paths:
        if filepath.endswith(fp) or fp.endswith(filepath):
            score += 2.0
            break
        if fp in filepath:
            score += 1.0

    for pn in partial_names:
        if basename == pn:
            score += 1.5
            break

    matched = 0
    for kw in keywords:
        if kw == stem:
            score += 0.5
            matched += 1
        elif kw in path_tokens:
            score += 0.3
            matched += 1

    if matched >= 2:
        score *= 1.3
    return score


def extract_features(
    filepath: str,
    position: int,
    total_predicted: int,
    issue_text: str,
    full_paths: Set[str],
    partial_names: Set[str],
    keywords: Set[str],
    base_predicted: Set[str],
    cochange_to_base: float,
    import_connected: bool,
    same_dir_as_base: bool,
    file_pr_count: int,
) -> List[float]:
    """Extract feature vector for a (issue, candidate_file) pair.

    Features:
    0: position_in_expanded (normalized)
    1: is_in_base_prediction
    2: text_relevance_score
    3: cochange_score_to_base
    4: import_connected_to_base
    5: same_dir_as_base
    6: file_pr_count (log-scaled)
    7: file_depth
    8: is_test_file
    9: is_init_file
    10: basename_length
    11: position_bucket (0=top5, 1=top10, 2=top20, 3=rest)
    """
    depth = filepath.count('/')
    basename = os.path.basename(filepath)
    is_test = 1.0 if ('test' in basename.lower() or '/test/' in filepath or '/tests/' in filepath) else 0.0
    is_init = 1.0 if basename == '__init__.py' else 0.0

    text_score = compute_text_score(filepath, full_paths, partial_names, keywords)

    if position < 5:
        pos_bucket = 0
    elif position < 10:
        pos_bucket = 1
    elif position < 20:
        pos_bucket = 2
    else:
        pos_bucket = 3

    return [
        position / max(total_predicted, 1),  # 0: normalized position
        1.0 if filepath in base_predicted else 0.0,  # 1: in base
        text_score,  # 2: text relevance
        cochange_to_base,  # 3: co-change score
        1.0 if import_connected else 0.0,  # 4: import connected
        1.0 if same_dir_as_base else 0.0,  # 5: same dir
        np.log1p(file_pr_count),  # 6: file frequency (log)
        depth,  # 7: depth
        is_test,  # 8: test file
        is_init,  # 9: init file
        len(basename),  # 10: basename length
        pos_bucket,  # 11: position bucket
    ]


FEATURE_NAMES = [
    'position_norm', 'in_base', 'text_score', 'cochange_to_base',
    'import_connected', 'same_dir', 'file_pr_count_log', 'depth',
    'is_test', 'is_init', 'basename_len', 'position_bucket',
]


# ========== Data Loading ==========

def build_file_cochange_scores(train_data: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build per-repo co-change score matrix.
    Returns {repo: {file_a: {file_b: score}}}
    """
    repo_pairs = defaultdict(Counter)
    repo_file_counts = defaultdict(Counter)
    for item in train_data:
        repo = item['repo']
        files = item.get('changed_py_files', [])
        for f in files:
            repo_file_counts[repo][f] += 1
        for i, fa in enumerate(sorted(files)):
            for fb in sorted(files)[i + 1:]:
                repo_pairs[repo][(fa, fb)] += 1

    result = defaultdict(lambda: defaultdict(dict))
    for repo, pairs in repo_pairs.items():
        for (fa, fb), count in pairs.items():
            min_count = min(repo_file_counts[repo][fa], repo_file_counts[repo][fb])
            score = count / max(min_count, 1)
            result[repo][fa][fb] = score
            result[repo][fb][fa] = score
    return result


def build_import_adjacency(dep_graph_dir: str) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-repo import adjacency. Returns {repo: {file: {neighbors}}}."""
    result = defaultdict(lambda: defaultdict(set))
    if not os.path.isdir(dep_graph_dir):
        return result
    for fname in os.listdir(dep_graph_dir):
        if not fname.endswith("_rels.json"):
            continue
        repo = fname.replace("_rels.json", "")
        with open(os.path.join(dep_graph_dir, fname)) as f:
            rels = json.load(f)
        for importer, imported_list in rels.get('file_imports', {}).items():
            for imported in imported_list:
                if importer.endswith('.py') and imported.endswith('.py'):
                    result[repo][importer].add(imported)
                    result[repo][imported].add(importer)
    return result


# ========== Commands ==========

def cmd_generate_data(args):
    """Generate training data for the reranker from expanded predictions."""
    print("Loading data...")
    with open(args.train_data) as f:
        train_data = [json.loads(l) for l in f]

    test_items = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_items[key] = item

    preds = {}
    with open(args.predictions) as f:
        for line in f:
            p = json.loads(line)
            key = (p['repo'], p['issue_id'])
            preds[key] = p

    # Also load base predictions if available
    base_preds = {}
    if args.base_predictions:
        with open(args.base_predictions) as f:
            for line in f:
                p = json.loads(line)
                key = (p['repo'], p['issue_id'])
                base_preds[key] = set(p['predicted'])

    print("Building structural indexes...")
    cochange_scores = build_file_cochange_scores(train_data)
    import_adj = build_import_adjacency(args.dep_graph_dir)

    # Load file trees for PR counts
    file_pr_counts = {}
    for fname in os.listdir(args.file_tree_dir):
        if fname.endswith('.json'):
            repo = fname.replace('.json', '')
            with open(os.path.join(args.file_tree_dir, fname)) as f:
                tree = json.load(f)
            file_pr_counts[repo] = tree.get('file_to_pr_count', {})

    print("Extracting features...")
    examples = []
    for key, test_item in test_items.items():
        if key not in preds:
            continue
        gt = set(test_item.get('changed_py_files', []))
        if not gt:
            continue

        pred = preds[key]
        predicted = pred['predicted']
        repo = key[0]
        issue_text = test_item.get('issue_text', '')
        full_paths, partial_names, keywords = extract_file_mentions(issue_text)
        base_pred_set = base_preds.get(key, set())
        repo_cochange = cochange_scores.get(repo, {})
        repo_imports = import_adj.get(repo, {})
        repo_pr = file_pr_counts.get(repo, {})

        for pos, filepath in enumerate(predicted):
            # Co-change score: max score to any base-predicted file
            max_cc = 0.0
            for bp in base_pred_set:
                cc = repo_cochange.get(filepath, {}).get(bp, 0.0)
                max_cc = max(max_cc, cc)

            # Import connected to any base prediction
            file_imports = repo_imports.get(filepath, set())
            import_conn = bool(file_imports & base_pred_set)

            # Same dir as any base prediction
            file_dir = os.path.dirname(filepath)
            same_dir = any(os.path.dirname(bp) == file_dir for bp in base_pred_set)

            features = extract_features(
                filepath=filepath,
                position=pos,
                total_predicted=len(predicted),
                issue_text=issue_text,
                full_paths=full_paths,
                partial_names=partial_names,
                keywords=keywords,
                base_predicted=base_pred_set,
                cochange_to_base=max_cc,
                import_connected=import_conn,
                same_dir_as_base=same_dir,
                file_pr_count=repo_pr.get(filepath, 0),
            )

            label = 1 if filepath in gt else 0
            examples.append({
                'features': features,
                'label': label,
                'repo': repo,
                'issue_id': key[1],
                'filepath': filepath,
                'position': pos,
            })

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    n_pos = sum(1 for ex in examples if ex['label'] == 1)
    n_neg = len(examples) - n_pos
    print(f"Generated {len(examples)} examples ({n_pos} positive, {n_neg} negative)")
    print(f"Positive rate: {100 * n_pos / len(examples):.2f}%")
    print(f"Saved to {args.output}")


def cmd_train(args):
    """Train the reranker model."""
    print(f"Loading training data from {args.train_data}...")
    examples = []
    with open(args.train_data) as f:
        for line in f:
            examples.append(json.loads(line))

    X = np.array([ex['features'] for ex in examples])
    y = np.array([ex['label'] for ex in examples])

    print(f"Training data: {len(examples)} examples, {y.sum()} positive ({100 * y.mean():.2f}%)")
    print(f"Feature shape: {X.shape}")

    try:
        import lightgbm as lgb
        print("Using LightGBM classifier")

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)

        # Feature importance
        importance = model.feature_importances_
        for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
            print(f"  {name:25s}: {imp:.0f}")

    except ImportError:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        print("LightGBM not available, using Logistic Regression")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = {
            'scaler': scaler,
            'classifier': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
            )
        }
        model['classifier'].fit(X_scaled, y)

        coefs = model['classifier'].coef_[0]
        for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -abs(x[1])):
            print(f"  {name:25s}: {coef:+.4f}")

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    with open(args.output_model, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output_model}")


def cmd_rerank(args):
    """Apply trained reranker to rerank expanded predictions."""
    print("Loading model...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    print("Loading data...")
    with open(args.train_data) as f:
        train_data = [json.loads(l) for l in f]

    test_items = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_items[key] = item

    preds = []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))

    base_preds = {}
    if args.base_predictions:
        with open(args.base_predictions) as f:
            for line in f:
                p = json.loads(line)
                key = (p['repo'], p['issue_id'])
                base_preds[key] = set(p['predicted'])

    cochange_scores = build_file_cochange_scores(train_data)
    import_adj = build_import_adjacency(args.dep_graph_dir)

    file_pr_counts = {}
    for fname in os.listdir(args.file_tree_dir):
        if fname.endswith('.json'):
            repo = fname.replace('.json', '')
            with open(os.path.join(args.file_tree_dir, fname)) as f:
                tree = json.load(f)
            file_pr_counts[repo] = tree.get('file_to_pr_count', {})

    print("Reranking predictions...")
    reranked = []
    for p in preds:
        repo = p['repo']
        issue_id = p['issue_id']
        key = (repo, issue_id)
        predicted = p['predicted']
        test_item = test_items.get(key, {})
        issue_text = test_item.get('issue_text', '')
        full_paths, partial_names, keywords = extract_file_mentions(issue_text)
        base_pred_set = base_preds.get(key, set())
        repo_cochange = cochange_scores.get(repo, {})
        repo_imports = import_adj.get(repo, {})
        repo_pr = file_pr_counts.get(repo, {})

        # Extract features for all candidates
        features = []
        for pos, filepath in enumerate(predicted):
            max_cc = 0.0
            for bp in base_pred_set:
                cc = repo_cochange.get(filepath, {}).get(bp, 0.0)
                max_cc = max(max_cc, cc)
            file_imports = repo_imports.get(filepath, set())
            import_conn = bool(file_imports & base_pred_set)
            file_dir = os.path.dirname(filepath)
            same_dir = any(os.path.dirname(bp) == file_dir for bp in base_pred_set)

            feat = extract_features(
                filepath=filepath,
                position=pos,
                total_predicted=len(predicted),
                issue_text=issue_text,
                full_paths=full_paths,
                partial_names=partial_names,
                keywords=keywords,
                base_predicted=base_pred_set,
                cochange_to_base=max_cc,
                import_connected=import_conn,
                same_dir_as_base=same_dir,
                file_pr_count=repo_pr.get(filepath, 0),
            )
            features.append(feat)

        if not features:
            reranked.append(p)
            continue

        X = np.array(features)

        # Get probability scores
        if isinstance(model, dict):
            # Logistic regression with scaler
            X_scaled = model['scaler'].transform(X)
            scores = model['classifier'].predict_proba(X_scaled)[:, 1]
        else:
            scores = model.predict_proba(X)[:, 1]

        # Rerank by model score, but keep top-1 fixed (highest confidence)
        scored_files = list(zip(predicted, scores))
        # Keep first file fixed, rerank the rest
        first_file = scored_files[0]
        rest = sorted(scored_files[1:], key=lambda x: -x[1])
        reranked_files = [first_file[0]] + [f for f, _ in rest]

        # Compute metrics
        gt = set(p.get('ground_truth', p.get('changed_py_files', [])))
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            topk = set(reranked_files[:k])
            hits = len(gt & topk)
            metrics[f'hit@{k}'] = (hits / len(gt)) * 100 if gt else 0.0

        new_p = dict(p)
        new_p['predicted'] = reranked_files
        new_p['metrics'] = metrics
        reranked.append(new_p)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for rp in reranked:
            f.write(json.dumps(rp) + '\n')

    # Summary
    overall = defaultdict(list)
    for rp in reranked:
        for k, v in rp['metrics'].items():
            overall[k].append(v)
    overall_means = {k: sum(v) / len(v) for k, v in overall.items()}

    summary = {'overall': overall_means}
    summary_path = args.output.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Reranked {len(reranked)} predictions")
    print(f"Hit@1={overall_means['hit@1']:.2f}% Hit@5={overall_means['hit@5']:.2f}% "
          f"Hit@10={overall_means['hit@10']:.2f}% Hit@20={overall_means['hit@20']:.2f}%")


def _train_model(X, y):
    """Train a reranker model on features X and labels y. Returns model."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, min_child_samples=20,
            class_weight='balanced', random_state=42,
            n_jobs=-1, verbose=-1,
        )
        model.fit(X, y)
        return model
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42,
        )
        clf.fit(X_scaled, y)
        return {'scaler': scaler, 'classifier': clf}


def _predict_scores(model, X):
    """Get probability scores from a trained reranker model."""
    if len(X) == 0:
        return np.array([])
    if isinstance(model, dict):
        X_scaled = model['scaler'].transform(X)
        return model['classifier'].predict_proba(X_scaled)[:, 1]
    return model.predict_proba(X)[:, 1]


def _extract_all_features(
    preds_list, test_items, base_preds, cochange_scores,
    import_adj, file_pr_counts,
):
    """Extract features for all predictions. Returns list of per-issue feature dicts."""
    per_issue = []
    for p in preds_list:
        repo = p['repo']
        issue_id = p['issue_id']
        key = (repo, issue_id)
        predicted = p['predicted']
        test_item = test_items.get(key, {})
        issue_text = test_item.get('issue_text', '')
        gt = set(test_item.get('changed_py_files', []))

        full_paths, partial_names, keywords = extract_file_mentions(issue_text)
        base_pred_set = base_preds.get(key, set())
        repo_cochange = cochange_scores.get(repo, {})
        repo_imports = import_adj.get(repo, {})
        repo_pr = file_pr_counts.get(repo, {})

        features = []
        labels = []
        for pos, filepath in enumerate(predicted):
            max_cc = 0.0
            for bp in base_pred_set:
                cc = repo_cochange.get(filepath, {}).get(bp, 0.0)
                max_cc = max(max_cc, cc)
            file_imports = repo_imports.get(filepath, set())
            import_conn = bool(file_imports & base_pred_set)
            file_dir = os.path.dirname(filepath)
            same_dir = any(os.path.dirname(bp) == file_dir for bp in base_pred_set)

            feat = extract_features(
                filepath=filepath,
                position=pos,
                total_predicted=len(predicted),
                issue_text=issue_text,
                full_paths=full_paths,
                partial_names=partial_names,
                keywords=keywords,
                base_predicted=base_pred_set,
                cochange_to_base=max_cc,
                import_connected=import_conn,
                same_dir_as_base=same_dir,
                file_pr_count=repo_pr.get(filepath, 0),
            )
            features.append(feat)
            labels.append(1 if filepath in gt else 0)

        per_issue.append({
            'key': key,
            'pred': p,
            'features': np.array(features) if features else np.empty((0, len(FEATURE_NAMES))),
            'labels': np.array(labels) if labels else np.empty(0),
            'predicted': predicted,
            'gt': gt,
        })
    return per_issue


def cmd_cv_rerank(args):
    """Leak-free 5-fold CV reranking: each example scored by model that never saw its label.

    Folds are stratified by repo to avoid repo-level leakage.
    """
    from sklearn.model_selection import KFold

    print(f"=== Leak-Free {args.n_folds}-Fold CV Reranking ===")
    print("Loading data...")

    with open(args.train_data) as f:
        train_data = [json.loads(l) for l in f]

    test_items = {}
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (item['repo'], item['issue_id'])
            test_items[key] = item

    preds_list = []
    with open(args.predictions) as f:
        for line in f:
            preds_list.append(json.loads(line))

    base_preds = {}
    if args.base_predictions:
        with open(args.base_predictions) as f:
            for line in f:
                p = json.loads(line)
                key = (p['repo'], p['issue_id'])
                base_preds[key] = set(p['predicted'])

    print("Building structural indexes...")
    cochange_scores = build_file_cochange_scores(train_data)
    import_adj = build_import_adjacency(args.dep_graph_dir)

    file_pr_counts = {}
    for fname in os.listdir(args.file_tree_dir):
        if fname.endswith('.json'):
            repo = fname.replace('.json', '')
            with open(os.path.join(args.file_tree_dir, fname)) as f:
                tree = json.load(f)
            file_pr_counts[repo] = tree.get('file_to_pr_count', {})

    print("Extracting features for all predictions...")
    per_issue = _extract_all_features(
        preds_list, test_items, base_preds,
        cochange_scores, import_adj, file_pr_counts,
    )

    # Assign fold indices — stratify by repo to avoid repo-level leakage
    repos = list(set(pi['key'][0] for pi in per_issue))
    repos.sort()
    repo_to_fold = {}
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    for fold_idx, (_, test_idx) in enumerate(kf.split(repos)):
        for i in test_idx:
            repo_to_fold[repos[i]] = fold_idx

    issue_folds = [repo_to_fold[pi['key'][0]] for pi in per_issue]

    print(f"Fold distribution: {Counter(issue_folds)}")

    # Cross-validation: for each fold, train on other folds, predict on this fold
    reranked_all = [None] * len(per_issue)
    fold_metrics = defaultdict(lambda: defaultdict(list))

    for fold in range(args.n_folds):
        # Collect training data from other folds
        train_X, train_y = [], []
        test_indices = []
        for i, pi in enumerate(per_issue):
            if issue_folds[i] == fold:
                test_indices.append(i)
            else:
                if len(pi['features']) > 0:
                    train_X.append(pi['features'])
                    train_y.append(pi['labels'])

        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)

        n_pos = train_y.sum()
        print(f"\nFold {fold}: train={len(train_y)} ({n_pos} pos), test={len(test_indices)} issues")

        # Train
        model = _train_model(train_X, train_y)

        # Print feature importance for last fold
        if fold == args.n_folds - 1:
            if isinstance(model, dict):
                coefs = model['classifier'].coef_[0]
                print("Feature coefficients (Logistic Regression):")
                for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -abs(x[1])):
                    print(f"  {name:25s}: {coef:+.4f}")
            else:
                print("Feature importance (LightGBM):")
                for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
                    print(f"  {name:25s}: {imp:.0f}")

        # Predict on held-out fold
        for i in test_indices:
            pi = per_issue[i]
            p = pi['pred']
            predicted = pi['predicted']

            if len(pi['features']) == 0:
                reranked_all[i] = p
                continue

            scores = _predict_scores(model, pi['features'])

            # Rerank: keep top-1 fixed, rerank rest by model score
            scored_files = list(zip(predicted, scores))
            first_file = scored_files[0]
            rest = sorted(scored_files[1:], key=lambda x: -x[1])
            reranked_files = [first_file[0]] + [f for f, _ in rest]

            gt = pi['gt']
            metrics = {}
            for k in [1, 3, 5, 10, 20]:
                topk = set(reranked_files[:k])
                hits = len(gt & topk)
                metrics[f'hit@{k}'] = (hits / len(gt)) * 100 if gt else 0.0

            new_p = dict(p)
            new_p['predicted'] = reranked_files
            new_p['metrics'] = metrics
            reranked_all[i] = new_p

            for k, v in metrics.items():
                fold_metrics[fold][k].append(v)

    # Aggregate results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for rp in reranked_all:
            f.write(json.dumps(rp) + '\n')

    overall = defaultdict(list)
    for rp in reranked_all:
        for k, v in rp['metrics'].items():
            overall[k].append(v)
    overall_means = {k: sum(v) / len(v) for k, v in overall.items()}

    summary = {
        'overall': overall_means,
        'method': f'{args.n_folds}-fold CV learned reranker (leak-free)',
        'n_folds': args.n_folds,
    }

    # Per-fold results
    print(f"\n{'='*60}")
    print(f"Per-fold results:")
    for fold in range(args.n_folds):
        fm = fold_metrics[fold]
        h1 = sum(fm['hit@1']) / len(fm['hit@1'])
        h5 = sum(fm['hit@5']) / len(fm['hit@5'])
        h10 = sum(fm['hit@10']) / len(fm['hit@10'])
        print(f"  Fold {fold}: H@1={h1:.2f} H@5={h5:.2f} H@10={h10:.2f} (n={len(fm['hit@1'])})")

    print(f"\n{'='*60}")
    print(f"Overall ({args.n_folds}-fold CV, LEAK-FREE):")
    print(f"  Hit@1={overall_means['hit@1']:.2f}% "
          f"Hit@5={overall_means['hit@5']:.2f}% "
          f"Hit@10={overall_means['hit@10']:.2f}% "
          f"Hit@20={overall_means['hit@20']:.2f}%")

    summary_path = args.output.replace('predictions.jsonl', 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Generate data
    gen = subparsers.add_parser('generate_data')
    gen.add_argument('--predictions', required=True)
    gen.add_argument('--base_predictions', default=None,
                     help='Base (pre-expansion) predictions for structural features')
    gen.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    gen.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    gen.add_argument('--dep_graph_dir', default='data/dep_graphs')
    gen.add_argument('--file_tree_dir', default='data/file_trees')
    gen.add_argument('--output', required=True)

    # Train
    tr = subparsers.add_parser('train')
    tr.add_argument('--train_data', required=True)
    tr.add_argument('--output_model', required=True)

    # Rerank
    rr = subparsers.add_parser('rerank')
    rr.add_argument('--predictions', required=True)
    rr.add_argument('--base_predictions', default=None)
    rr.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    rr.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    rr.add_argument('--dep_graph_dir', default='data/dep_graphs')
    rr.add_argument('--file_tree_dir', default='data/file_trees')
    rr.add_argument('--model', required=True)
    rr.add_argument('--output', required=True)

    # CV Rerank (leak-free)
    cv = subparsers.add_parser('cv_rerank',
                               help='Leak-free k-fold CV reranking (recommended)')
    cv.add_argument('--predictions', required=True)
    cv.add_argument('--base_predictions', default=None)
    cv.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    cv.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    cv.add_argument('--dep_graph_dir', default='data/dep_graphs')
    cv.add_argument('--file_tree_dir', default='data/file_trees')
    cv.add_argument('--output', required=True)
    cv.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'generate_data':
        cmd_generate_data(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'rerank':
        cmd_rerank(args)
    elif args.command == 'cv_rerank':
        cmd_cv_rerank(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
