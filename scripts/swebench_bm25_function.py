#!/usr/bin/env python3
"""
Function-level BM25 for SWE-bench Lite.

Indexes individual functions/classes/methods from Python files.
File-level score = max function-level score in that file.
This matches the approach used by LocAgent and other SWE-bench papers.

Usage:
    python scripts/swebench_bm25_function.py \
        --repos_dir data/swebench_lite/repos \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --output data/rankft/swebench_bm25_function_top500.jsonl
"""
import ast
import os
import re
import json
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def extract_code_entities(filepath: str) -> List[Tuple[str, str, str]]:
    """Extract functions, classes, and methods from a Python file.

    Returns list of (entity_name, entity_type, entity_code).
    """
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        # Fallback: return whole file as one entity
        try:
            with open(filepath, 'r', errors='replace') as f:
                lines = f.readlines()[:100]
            return [("module", "module", "".join(lines))]
        except:
            return []

    source_lines = source.split('\n')
    entities = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            try:
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 20
                code = '\n'.join(source_lines[start:end])
                entities.append((name, "function", code))
            except:
                entities.append((name, "function", ""))
        elif isinstance(node, ast.ClassDef):
            name = node.name
            try:
                start = node.lineno - 1
                # Just get class header + first few lines (not entire class)
                end = min(node.lineno + 30, node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno + 30)
                code = '\n'.join(source_lines[start:end])
                entities.append((name, "class", code))
            except:
                entities.append((name, "class", ""))

    # If no entities found, use whole file (first 100 lines)
    if not entities:
        entities.append(("module", "module", '\n'.join(source_lines[:100])))

    return entities


def tokenize_code(text: str) -> List[str]:
    """Tokenize code for BM25."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'[_/\-.]', ' ', text)
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    stopwords = {
        'the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'had',
        'can', 'may', 'use', 'def', 'class', 'self', 'return', 'import',
        'from', 'if', 'else', 'elif', 'try', 'except', 'with', 'as',
        'in', 'is', 'or', 'none', 'true', 'false', 'pass', 'raise',
    }
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def tokenize_query(text: str) -> List[str]:
    """Tokenize query (issue text)."""
    tokens = tokenize_code(text)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        parts = re.split(r'[/_\-.]', ref.replace('.py', ''))
        tokens.extend([p.lower() for p in parts if len(p) > 1] * 3)
    quoted = re.findall(r'[`\'"](\w+)[`\'"]', text)
    tokens.extend([q.lower() for q in quoted if len(q) > 1] * 2)
    dotted = re.findall(r'\b\w+(?:\.\w+){2,}\b', text)
    for d in dotted:
        tokens.extend(d.lower().split('.'))
    return tokens


def build_function_index(repos_dir: str, repo_name: str):
    """Build function-level BM25 index for a repo.

    Returns: (bm25, entity_list) where entity_list[i] = (filepath, entity_name, entity_type)
    """
    repo_dir = os.path.join(repos_dir, repo_name)
    if not os.path.isdir(repo_dir):
        return None, []

    py_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, repo_dir)
                py_files.append(rel_path)

    # Extract entities from all files
    all_tokens = []
    entity_info = []  # (filepath, name, type)

    for fp in py_files:
        full_path = os.path.join(repo_dir, fp)
        entities = extract_code_entities(full_path)

        for name, etype, code in entities:
            # Document = path tokens + entity name + entity code
            path_tokens = re.split(r'[/_\-.]', fp.replace('.py', ''))
            path_tokens = [t.lower() for t in path_tokens if len(t) > 1]

            name_tokens = tokenize_code(name)
            code_tokens = tokenize_code(code)

            # Weight: path 3x, name 5x, code 1x
            doc_tokens = path_tokens * 3 + name_tokens * 5 + code_tokens
            if doc_tokens:
                all_tokens.append(doc_tokens)
                entity_info.append((fp, name, etype))

    if not all_tokens:
        return None, []

    bm25 = BM25Okapi(all_tokens)
    return bm25, entity_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_bm25_function_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print('=== Function-Level BM25 for SWE-bench Lite ===')

    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print('  %d test examples' % len(examples))

    repo_examples = defaultdict(list)
    for ex in examples:
        repo_examples[ex['repo']].append(ex)

    results = []
    t0 = time.time()

    for repo_name, exs in sorted(repo_examples.items()):
        repo_t0 = time.time()
        bm25, entity_info = build_function_index(args.repos_dir, repo_name)
        build_time = time.time() - repo_t0

        if bm25 is None:
            print('  %s: no entities found' % repo_name)
            continue

        n_entities = len(entity_info)
        n_files = len(set(e[0] for e in entity_info))
        print('  %s: %d entities in %d files (%.1fs)' % (repo_name, n_entities, n_files, build_time))

        for ex in exs:
            issue_text = ex.get('issue_text', '')
            gt = set(ex.get('changed_py_files', []))

            query_tokens = tokenize_query(issue_text)
            scores = bm25.get_scores(query_tokens)

            # Aggregate to file level: max entity score per file
            file_scores = defaultdict(float)
            for i, (fp, name, etype) in enumerate(entity_info):
                file_scores[fp] = max(file_scores[fp], scores[i])

            # Sort by score
            ranked_files = sorted(file_scores.items(), key=lambda x: -x[1])
            candidates = [f for f, _ in ranked_files[:args.top_k]]

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

    # Metrics
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
