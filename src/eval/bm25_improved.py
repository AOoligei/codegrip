"""
Improved BM25 baseline with better tokenization.
Key improvements over bm25_baseline.py:
1. Double-index: both full path and basename tokens
2. Extract identifiers from issue text (class names, function names)
3. Query expansion with file path references found in issue
4. Separate scoring for directory structure matching

Usage:
    python src/eval/bm25_improved.py \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --file_tree_dir data/swebench_lite/file_trees \
        --output_dir experiments/baselines/swebench_bm25_improved
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


def tokenize_path_improved(path: str) -> List[str]:
    """Improved path tokenization with both full path and basename tokens."""
    tokens = []
    
    # Remove .py extension
    clean = re.sub(r'\.py$', '', path)
    
    # Full path tokens (split on / and _)
    parts = re.split(r'[/_\-.]', clean)
    for part in parts:
        # Split camelCase
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        sub_tokens = sub.lower().split()
        tokens.extend(sub_tokens)
    
    # Basename tokens (weighted by repeating)
    basename = os.path.basename(clean)
    basename_parts = re.split(r'[_\-.]', basename)
    for part in basename_parts:
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        sub_tokens = sub.lower().split()
        tokens.extend(sub_tokens)  # Basename appears twice = higher weight
    
    # Directory name tokens
    dirname = os.path.dirname(clean)
    if dirname:
        dir_parts = re.split(r'[/_\-.]', dirname)
        for part in dir_parts:
            sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            tokens.extend(sub.lower().split())
    
    # Full path as a single token (for exact matches)
    tokens.append(clean.replace('/', '.').lower())
    
    return [t for t in tokens if len(t) > 1]


def tokenize_issue_improved(text: str) -> List[str]:
    """Improved issue tokenization with identifier extraction."""
    tokens = []
    text_lower = text.lower()
    
    # Standard tokenization
    basic_tokens = re.findall(r'[a-z_][a-z0-9_]*', text_lower)
    tokens.extend(basic_tokens)
    
    # Extract CamelCase identifiers and split
    camel_words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    for w in camel_words:
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', w)
        sub_tokens = sub.lower().split()
        tokens.extend(sub_tokens)
        # Also add full identifier
        tokens.append(w.lower())
    
    # Extract file path references (weighted higher)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        ref_tokens = tokenize_path_improved(ref)
        tokens.extend(ref_tokens * 2)  # Double weight for explicit path refs
    
    # Extract dotted names (e.g., django.db.models)
    dotted = re.findall(r'\b\w+(?:\.\w+){2,}\b', text)
    for d in dotted:
        parts = d.lower().split('.')
        tokens.extend(parts)
    
    # Extract quoted identifiers
    quoted = re.findall(r'[`\'"](\w+)[`\'"]', text)
    tokens.extend([q.lower() for q in quoted if len(q) > 1])
    
    # Remove stopwords
    stop = {'this', 'that', 'with', 'from', 'have', 'been', 'when', 'what',
            'does', 'should', 'could', 'would', 'will', 'than', 'then',
            'also', 'into', 'some', 'other', 'more', 'after', 'before',
            'title', 'description', 'the', 'and', 'for', 'not', 'but',
            'are', 'was', 'were', 'has', 'had', 'can', 'may', 'use',
            'using', 'used', 'like', 'just', 'about', 'get', 'set'}
    
    return [t for t in tokens if len(t) > 1 and t not in stop]


def compute_hit_at_k(predicted: List[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--file_tree_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    test_data = []
    with open(args.test_data) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('changed_py_files'):
                test_data.append(ex)
    
    file_trees = {}
    for fp in Path(args.file_tree_dir).glob('*.json'):
        with open(fp) as f:
            tree = json.load(f)
            file_trees[tree['repo']] = tree
    
    print(f"Loaded {len(test_data)} test examples, {len(file_trees)} file trees")
    
    os.makedirs(args.output_dir, exist_ok=True)
    predictions = []
    all_metrics = {f'hit@{k}': [] for k in [1, 3, 5, 10, 20]}
    recall_at_k = {k: [] for k in [20, 50, 100, 200, 500]}
    
    start = time.time()
    for ex in test_data:
        repo = ex['repo']
        gt = set(ex['changed_py_files'])
        
        if repo not in file_trees:
            continue
        candidates = file_trees[repo].get('py_files', [])
        if not candidates:
            continue
        
        tokenized = [tokenize_path_improved(c) for c in candidates]
        bm25 = BM25Okapi(tokenized)
        query = tokenize_issue_improved(ex['issue_text'])
        
        if not query:
            scores = np.zeros(len(candidates))
        else:
            scores = bm25.get_scores(query)
        
        ranked_idx = np.argsort(scores)[::-1]
        
        # Full ranking for recall computation
        full_ranked = [candidates[i] for i in ranked_idx]
        
        # Top-20 for hit metrics
        ranked = full_ranked[:20]
        
        metrics = {}
        for k in [1, 3, 5, 10, 20]:
            h = compute_hit_at_k(ranked, gt, k)
            metrics[f'hit@{k}'] = h
            all_metrics[f'hit@{k}'].append(h)
        
        # Recall at various K
        for k in [20, 50, 100, 200, 500]:
            hit = 1.0 if gt & set(full_ranked[:k]) else 0.0
            recall_at_k[k].append(hit)
        
        predictions.append({
            'repo': repo,
            'issue_id': ex['issue_id'],
            'ground_truth': list(gt),
            'predicted': full_ranked[:500],  # Store top-500 for reranking
            'metrics': metrics,
            'method': 'bm25_improved',
        })
    
    elapsed = time.time() - start
    
    pred_path = os.path.join(args.output_dir, 'predictions.jsonl')
    with open(pred_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')
    
    summary = {
        'method': 'bm25_improved',
        'num_examples': len(predictions),
        'wall_clock_seconds': round(elapsed, 2),
        'metrics': {k: round(np.mean(v) * 100, 2) for k, v in all_metrics.items()},
        'recall': {f'recall@{k}': round(np.mean(v) * 100, 2) for k, v in recall_at_k.items()},
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nImproved BM25 results ({len(predictions)} examples, {elapsed:.1f}s):")
    for k, v in summary['metrics'].items():
        print(f"  {k}: {v}")
    print(f"\nRecall@K:")
    for k, v in summary['recall'].items():
        print(f"  {k}: {v}")
    
    return summary


if __name__ == '__main__':
    main()
