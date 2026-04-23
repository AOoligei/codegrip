#!/usr/bin/env python3
"""
Function-level BM25 for SWE-bench Lite — without test files.

Test file exclusion gave +9% Acc@1 for chunk BM25. Apply to function-level too.
Also adds Snowball stemmer.
"""
import ast
import os
import re
import json
import argparse
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
    def stem_tokens(tokens):
        return stemmer.stemWords(tokens)
    HAS_STEMMER = True
except ImportError:
    def stem_tokens(tokens):
        return tokens
    HAS_STEMMER = False


def is_test_file(filepath: str) -> bool:
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


def extract_code_entities(filepath: str) -> List[Tuple[str, str, str]]:
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError, ValueError):
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
                end = min(node.lineno + 30, node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno + 30)
                code = '\n'.join(source_lines[start:end])
                entities.append((name, "class", code))
            except:
                entities.append((name, "class", ""))

    if not entities:
        entities.append(("module", "module", '\n'.join(source_lines[:100])))

    return entities


def tokenize_code(text: str) -> List[str]:
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
    tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]
    if HAS_STEMMER:
        tokens = stem_tokens(tokens)
    return tokens


def tokenize_query(text: str) -> List[str]:
    tokens = tokenize_code(text)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        parts = re.split(r'[/_\-.]', ref.replace('.py', ''))
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        tokens.extend(extra * 3)
    quoted = re.findall(r'[`\'"]([\w.]+)[`\'"]', text)
    for q in quoted:
        parts = re.split(r'[._]', q)
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        tokens.extend(extra * 2)
    dotted = re.findall(r'\b\w+(?:\.\w+){2,}\b', text)
    for d in dotted:
        parts = d.lower().split('.')
        extra = [p for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        tokens.extend(extra)
    return tokens


def build_function_index(repos_dir, repo_name, exclude_tests=True):
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
                if exclude_tests and is_test_file(rel_path):
                    continue
                py_files.append(rel_path)

    all_tokens = []
    entity_info = []

    for fp in py_files:
        full_path = os.path.join(repo_dir, fp)
        entities = extract_code_entities(full_path)

        for name, etype, code in entities:
            path_tokens = re.split(r'[/_\-.]', fp.replace('.py', ''))
            path_tokens = [t.lower() for t in path_tokens if len(t) > 1]
            if HAS_STEMMER:
                path_tokens = stem_tokens(path_tokens)

            name_tokens = tokenize_code(name)
            code_tokens = tokenize_code(code)

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
    parser.add_argument('--output', default='data/rankft/swebench_bm25_function_notest_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print('=== Function-Level BM25 (no test files, with stemmer) ===')
    print(f'  Stemmer: {"Snowball" if HAS_STEMMER else "None"}')

    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f'  {len(examples)} test examples')

    repo_examples = defaultdict(list)
    for ex in examples:
        repo_examples[ex['repo']].append(ex)

    results = []
    t0 = time.time()

    for repo_name, exs in sorted(repo_examples.items()):
        bm25, entity_info = build_function_index(args.repos_dir, repo_name, exclude_tests=True)
        if bm25 is None:
            continue

        n_entities = len(entity_info)
        n_files = len(set(e[0] for e in entity_info))

        for ex in exs:
            issue_text = ex.get('issue_text', '')
            gt = set(ex.get('changed_py_files', []))
            query_tokens = tokenize_query(issue_text)
            scores = bm25.get_scores(query_tokens)

            file_scores = defaultdict(float)
            for i, (fp, name, etype) in enumerate(entity_info):
                file_scores[fp] = max(file_scores[fp], scores[i])

            ranked_files = sorted(file_scores.items(), key=lambda x: -x[1])
            candidates = [f for f, _ in ranked_files[:args.top_k]]

            results.append({
                'repo': repo_name,
                'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
                'bm25_candidates': candidates,
                'ground_truth': list(gt),
            })

    elapsed = time.time() - t0
    print(f'Processed {len(results)} examples in {elapsed:.1f}s')

    # Metrics
    k_values = [1, 3, 5, 10, 20, 50, 100, 200, 500]
    for k in k_values:
        acc = np.mean([
            1.0 if set(r['ground_truth']).issubset(set(r['bm25_candidates'][:k]))
            else 0.0
            for r in results
        ]) * 100
        print(f'  Acc@{k}: {acc:.2f}%')

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'Saved to {args.output}')

    # Also compute matched 274
    excluded = {
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
    matched = [r for r in results if r['issue_id'] not in excluded]
    print(f'\nMatched 274 protocol ({len(matched)} instances):')
    for k in [1, 3, 5, 10, 20]:
        acc = np.mean([
            1.0 if set(r['ground_truth']).issubset(set(r['bm25_candidates'][:k]))
            else 0.0
            for r in matched
        ]) * 100
        print(f'  Acc@{k}: {acc:.2f}%')


if __name__ == '__main__':
    main()
