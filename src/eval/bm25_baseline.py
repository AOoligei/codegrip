"""
BM25 + TF-IDF + Frequency baselines for GREPO file-level localization.
Provides strong non-neural retrieval baselines for fair comparison.

Usage:
    python src/eval/bm25_baseline.py \
        --test_data data/grepo_text/grepo_test.jsonl \
        --file_tree_dir data/file_trees \
        --output_dir experiments/bm25_baseline
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_path(path: str) -> List[str]:
    """Tokenize a file path into meaningful tokens.
    e.g., 'cirq/value/linear_dict.py' -> ['cirq', 'value', 'linear', 'dict']
    """
    # Remove extension
    path = re.sub(r'\.py$', '', path)
    # Split on / and _
    parts = re.split(r'[/_]', path)
    # Split camelCase
    tokens = []
    for part in parts:
        # Split camelCase: 'LinearDict' -> ['linear', 'dict']
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        tokens.extend(sub.lower().split())
    # Filter empty and very short tokens
    return [t for t in tokens if len(t) > 1]


def tokenize_issue(text: str) -> List[str]:
    """Tokenize issue text into tokens."""
    text = text.lower()
    # Remove common prefixes
    text = re.sub(r'^title:\s*', '', text)
    # Tokenize
    tokens = re.findall(r'[a-z_][a-z0-9_]*', text)
    return [t for t in tokens if len(t) > 1]


def compute_hit_at_k(predicted: List[str], ground_truth: Set[str], k: int) -> float:
    """Compute Hit@K metric."""
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def load_data(test_path: str, file_tree_dir: str, train_path: str = None):
    """Load test data, file trees, and optionally training data."""
    # Load test data
    test_data = []
    with open(test_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('changed_py_files'):
                test_data.append(ex)

    # Load file trees
    file_trees = {}
    for fp in Path(file_tree_dir).glob('*.json'):
        with open(fp) as f:
            tree = json.load(f)
            file_trees[tree['repo']] = tree

    # Load training data for frequency baseline
    train_data = []
    if train_path and os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                train_data.append(json.loads(line))

    return test_data, file_trees, train_data


def build_file_freq(train_data: List[dict]) -> Dict[str, Dict[str, int]]:
    """Build per-repo file change frequency from training data."""
    freq = defaultdict(lambda: defaultdict(int))
    for ex in train_data:
        repo = ex['repo']
        for f in ex.get('changed_py_files', []):
            freq[repo][f] += 1
    return freq


def run_bm25_baseline(test_data, file_trees, output_dir, method='bm25'):
    """Run BM25 retrieval on file paths."""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}

    start = time.time()
    for ex in test_data:
        repo = ex['repo']
        gt = set(ex['changed_py_files'])

        if repo not in file_trees:
            continue

        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue

        # Tokenize candidates
        tokenized_candidates = [tokenize_path(c) for c in candidates]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_candidates)

        # Tokenize query
        query_tokens = tokenize_issue(ex['issue_text'])
        if not query_tokens:
            ranked = candidates[:20]
        else:
            scores = bm25.get_scores(query_tokens)
            ranked_idx = np.argsort(scores)[::-1]
            ranked = [candidates[i] for i in ranked_idx[:20]]

        # Compute metrics
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)

        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': ranked,
            'metrics': metrics,
            'method': method,
        })

    elapsed = time.time() - start

    # Save predictions
    pred_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    # Compute and save summary
    summary = {
        'method': method,
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def run_frequency_baseline(test_data, file_trees, train_data, output_dir):
    """Baseline: rank files by historical change frequency."""
    os.makedirs(output_dir, exist_ok=True)
    file_freq = build_file_freq(train_data)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}

    start = time.time()
    for ex in test_data:
        repo = ex['repo']
        gt = set(ex['changed_py_files'])

        if repo not in file_trees:
            continue

        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue

        # Score by frequency
        freq = file_freq.get(repo, {})
        scored = [(c, freq.get(c, 0)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        ranked = [c for c, _ in scored[:20]]

        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)

        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': ranked,
            'metrics': metrics,
            'method': 'frequency',
        })

    elapsed = time.time() - start

    pred_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    summary = {
        'method': 'frequency',
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def run_tfidf_baseline(test_data, file_trees, output_dir):
    """TF-IDF cosine similarity baseline on file paths."""
    os.makedirs(output_dir, exist_ok=True)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}

    start = time.time()
    for ex in test_data:
        repo = ex['repo']
        gt = set(ex['changed_py_files'])

        if repo not in file_trees:
            continue

        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue

        # Create "documents" from file paths
        docs = [' '.join(tokenize_path(c)) for c in candidates]
        query = ' '.join(tokenize_issue(ex['issue_text']))

        if not query.strip():
            ranked = candidates[:20]
        else:
            # TF-IDF + cosine similarity
            try:
                vectorizer = TfidfVectorizer()
                all_docs = docs + [query]
                tfidf_matrix = vectorizer.fit_transform(all_docs)
                query_vec = tfidf_matrix[-1]
                doc_vecs = tfidf_matrix[:-1]
                scores = (doc_vecs @ query_vec.T).toarray().flatten()
                ranked_idx = np.argsort(scores)[::-1]
                ranked = [candidates[i] for i in ranked_idx[:20]]
            except ValueError:
                ranked = candidates[:20]

        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)

        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': ranked,
            'metrics': metrics,
            'method': 'tfidf',
        })

    elapsed = time.time() - start

    pred_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    summary = {
        'method': 'tfidf',
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def run_combined_baseline(test_data, file_trees, train_data, output_dir,
                          w_bm25=0.4, w_tfidf=0.3, w_freq=0.3):
    """Combined baseline: weighted sum of BM25 + TF-IDF + frequency scores."""
    os.makedirs(output_dir, exist_ok=True)
    file_freq = build_file_freq(train_data)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}

    start = time.time()
    for ex in test_data:
        repo = ex['repo']
        gt = set(ex['changed_py_files'])

        if repo not in file_trees:
            continue

        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue

        n = len(candidates)
        tokenized_candidates = [tokenize_path(c) for c in candidates]
        query_tokens = tokenize_issue(ex['issue_text'])

        # BM25 scores
        bm25 = BM25Okapi(tokenized_candidates)
        bm25_scores = bm25.get_scores(query_tokens) if query_tokens else np.zeros(n)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # TF-IDF scores
        docs = [' '.join(toks) for toks in tokenized_candidates]
        query_str = ' '.join(query_tokens)
        if query_str.strip():
            try:
                vectorizer = TfidfVectorizer()
                all_docs = docs + [query_str]
                tfidf_matrix = vectorizer.fit_transform(all_docs)
                tfidf_scores = (tfidf_matrix[:-1] @ tfidf_matrix[-1].T).toarray().flatten()
            except ValueError:
                tfidf_scores = np.zeros(n)
        else:
            tfidf_scores = np.zeros(n)
        tfidf_max = tfidf_scores.max() if tfidf_scores.max() > 0 else 1.0
        tfidf_norm = tfidf_scores / tfidf_max

        # Frequency scores
        freq = file_freq.get(repo, {})
        freq_scores = np.array([freq.get(c, 0) for c in candidates], dtype=float)
        freq_max = freq_scores.max() if freq_scores.max() > 0 else 1.0
        freq_norm = freq_scores / freq_max

        # Combined
        combined = w_bm25 * bm25_norm + w_tfidf * tfidf_norm + w_freq * freq_norm
        ranked_idx = np.argsort(combined)[::-1]
        ranked = [candidates[i] for i in ranked_idx[:20]]

        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)

        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': ranked,
            'metrics': metrics,
            'method': 'combined',
        })

    elapsed = time.time() - start

    pred_path = os.path.join(output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    summary = {
        'method': 'combined',
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
        'weights': {'bm25': w_bm25, 'tfidf': w_tfidf, 'freq': w_freq},
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--file_tree_dir', default='data/file_trees')
    parser.add_argument('--output_dir', default='experiments/baselines')
    args = parser.parse_args()

    print("Loading data...")
    test_data, file_trees, train_data = load_data(
        args.test_data, args.file_tree_dir, args.train_data
    )
    print(f"  Test: {len(test_data)} examples, File trees: {len(file_trees)} repos, "
          f"Train: {len(train_data)} examples")

    # Run all baselines
    print("\n=== BM25 (file path) ===")
    s = run_bm25_baseline(
        test_data, file_trees,
        os.path.join(args.output_dir, 'bm25_path')
    )
    print(f"  {s['metrics']}  ({s['wall_clock_seconds']}s)")

    print("\n=== TF-IDF (file path) ===")
    s = run_tfidf_baseline(
        test_data, file_trees,
        os.path.join(args.output_dir, 'tfidf_path')
    )
    print(f"  {s['metrics']}  ({s['wall_clock_seconds']}s)")

    print("\n=== Frequency (train history) ===")
    s = run_frequency_baseline(
        test_data, file_trees, train_data,
        os.path.join(args.output_dir, 'frequency')
    )
    print(f"  {s['metrics']}  ({s['wall_clock_seconds']}s)")

    print("\n=== Combined (BM25 + TF-IDF + Freq) ===")
    s = run_combined_baseline(
        test_data, file_trees, train_data,
        os.path.join(args.output_dir, 'combined')
    )
    print(f"  {s['metrics']}  ({s['wall_clock_seconds']}s)")

    print("\nDone! All baselines saved to", args.output_dir)


if __name__ == '__main__':
    main()
