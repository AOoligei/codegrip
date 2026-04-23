#!/usr/bin/env python3
"""
Content-based BM25 retrieval for SWE-bench Lite.

Unlike path-only BM25, this indexes file CONTENT (first N lines of each .py file)
and searches with the full issue text. This matches the standard BM25 baseline
used in LocAgent, SweRank, and other SWE-bench localization papers.

Usage:
    python scripts/swebench_bm25_content.py \
        --repos_dir data/swebench_lite/repos \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --output data/rankft/swebench_bm25_content_top500.jsonl \
        --top_k 500 --max_lines 200
"""
import os
import re
import json
import argparse
import time
from collections import defaultdict
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi


def read_file_content(filepath: str, max_lines: int = 200) -> str:
    """Read first max_lines of a file."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        return '\n'.join(lines)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return ''


def tokenize_code(text: str) -> List[str]:
    """Tokenize code/text for BM25.
    Splits on whitespace, punctuation, camelCase, snake_case.
    """
    # Split camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    # Split snake_case and path separators
    text = re.sub(r'[_/\-.]', ' ', text)
    # Extract tokens
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    # Remove very short tokens and common keywords
    stopwords = {
        'the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'had',
        'can', 'may', 'use', 'def', 'class', 'self', 'return', 'import',
        'from', 'if', 'else', 'elif', 'try', 'except', 'with', 'as',
        'in', 'is', 'or', 'none', 'true', 'false', 'pass', 'raise',
        'this', 'that', 'will', 'would', 'should', 'could',
    }
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def tokenize_path(path: str) -> List[str]:
    """Tokenize a file path."""
    clean = re.sub(r'\.py$', '', path)
    parts = re.split(r'[/_\-.]', clean)
    tokens = []
    for part in parts:
        sub = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        tokens.extend(sub.lower().split())
    return [t for t in tokens if len(t) > 1]


def tokenize_document(path: str, content: str) -> List[str]:
    """Tokenize a document (path + content) for BM25 indexing."""
    # Path tokens (weighted 3x for emphasis)
    path_tokens = tokenize_path(path) * 3
    # Content tokens
    content_tokens = tokenize_code(content)
    return path_tokens + content_tokens


def tokenize_query(text: str) -> List[str]:
    """Tokenize a query (issue text)."""
    tokens = tokenize_code(text)
    # Extract file path references (weighted higher)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        tokens.extend(tokenize_path(ref) * 3)
    # Extract quoted identifiers
    quoted = re.findall(r'[`\'"](\w+)[`\'"]', text)
    tokens.extend([q.lower() for q in quoted if len(q) > 1])
    # Extract dotted names
    dotted = re.findall(r'\b\w+(?:\.\w+){2,}\b', text)
    for d in dotted:
        tokens.extend(d.lower().split('.'))
    return tokens


def build_repo_index(repos_dir: str, repo_name: str, max_lines: int = 200):
    """Build BM25 index for all .py files in a repo."""
    repo_dir = os.path.join(repos_dir, repo_name)
    if not os.path.isdir(repo_dir):
        return None, []

    py_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Skip hidden dirs and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, repo_dir)
                py_files.append(rel_path)

    if not py_files:
        return None, []

    # Tokenize all files
    tokenized_docs = []
    valid_files = []
    for fp in py_files:
        content = read_file_content(os.path.join(repo_dir, fp), max_lines)
        tokens = tokenize_document(fp, content)
        if tokens:
            tokenized_docs.append(tokens)
            valid_files.append(fp)

    if not tokenized_docs:
        return None, []

    bm25 = BM25Okapi(tokenized_docs)
    return bm25, valid_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_bm25_content_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    parser.add_argument('--max_lines', type=int, default=200)
    args = parser.parse_args()

    print('=== Content-Based BM25 for SWE-bench Lite ===')

    # Load test data
    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print('  %d test examples' % len(examples))

    # Group by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo_examples[ex['repo']].append(ex)
    print('  %d repos' % len(repo_examples))

    # Process per repo (build index once, reuse)
    results = []
    t0 = time.time()

    for repo_name, exs in sorted(repo_examples.items()):
        repo_t0 = time.time()
        bm25, valid_files = build_repo_index(args.repos_dir, repo_name, args.max_lines)
        build_time = time.time() - repo_t0

        if bm25 is None:
            print('  %s: no files found, skipping' % repo_name)
            continue

        print('  %s: %d py files, index built in %.1fs' % (repo_name, len(valid_files), build_time))

        for ex in exs:
            issue_text = ex.get('issue_text', '')
            gt = set(ex.get('changed_py_files', []))

            query_tokens = tokenize_query(issue_text)
            scores = bm25.get_scores(query_tokens)

            # Get top-K
            top_indices = np.argsort(scores)[::-1][:args.top_k]
            candidates = [valid_files[i] for i in top_indices]

            gt_in_cands = bool(gt & set(candidates))

            results.append({
                'repo': repo_name,
                'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
                'bm25_candidates': candidates,
                'ground_truth': list(gt),
                'gt_in_candidates': gt_in_cands,
            })

    elapsed = time.time() - t0
    print('\nProcessed %d examples in %.1fs' % (len(results), elapsed))

    # Compute metrics
    k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500]
    for k in k_values:
        acc = np.mean([
            1.0 if set(r['ground_truth']).issubset(set(r['bm25_candidates'][:k]))
            else 0.0
            for r in results
        ]) * 100
        print('  Acc@%d: %.2f%%' % (k, acc))

    gt_coverage = np.mean([1.0 if r['gt_in_candidates'] else 0.0 for r in results]) * 100
    print('  GT in top-%d: %.1f%%' % (args.top_k, gt_coverage))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print('\nSaved to %s' % args.output)


if __name__ == '__main__':
    main()
