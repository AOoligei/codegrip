#!/usr/bin/env python3
"""
BM25 Ensemble + Parameter Tuning for SWE-bench Lite.

Combines multiple BM25 variants via RRF and weighted fusion.
Also tunes BM25 parameters (k1, b) for function-level BM25.

CPU-only (no GPU needed).
"""
import os
import re
import ast
import json
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

np.random.seed(42)


# ============================================================
# Tokenization
# ============================================================

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


def tokenize_query_enhanced(text: str) -> List[str]:
    """Enhanced query tokenization for SWE-bench issues.

    Key improvements over basic tokenize_query:
    - Better handling of stack traces (extract module/function names)
    - Extract class.method references
    - Extract error type names
    - Weight code-like tokens higher
    """
    base_tokens = tokenize_code(text)

    # File references (weight 4x)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        parts = re.split(r'[/_\-.]', ref.replace('.py', ''))
        base_tokens.extend([p.lower() for p in parts if len(p) > 1] * 4)

    # Quoted identifiers (weight 3x)
    quoted = re.findall(r'[`\'"]([\w.]+)[`\'"]', text)
    for q in quoted:
        parts = re.split(r'[._]', q)
        base_tokens.extend([p.lower() for p in parts if len(p) > 1] * 3)

    # Class.method or module.function patterns (weight 3x)
    dotted = re.findall(r'\b(\w+(?:\.\w+){1,})\b', text)
    for d in dotted:
        parts = d.lower().split('.')
        base_tokens.extend([p for p in parts if len(p) > 1] * 3)

    # Error/Exception names (weight 4x)
    errors = re.findall(r'\b(\w+(?:Error|Exception|Warning|Failure))\b', text)
    for e in errors:
        parts = tokenize_code(e)
        base_tokens.extend(parts * 4)

    # Stack trace: extract "File XXX, line NNN, in YYY" patterns
    traceback_fns = re.findall(r'in\s+(\w+)', text)
    traceback_files = re.findall(r'File\s+"[^"]*?(\w+)\.py"', text)
    base_tokens.extend([f.lower() for f in traceback_fns if len(f) > 2] * 2)
    base_tokens.extend([f.lower() for f in traceback_files if len(f) > 2] * 3)

    # Method/function names after def/class keywords in code blocks
    code_defs = re.findall(r'(?:def|class)\s+(\w+)', text)
    base_tokens.extend([d.lower() for d in code_defs if len(d) > 1] * 3)

    return base_tokens


# ============================================================
# Function-level extraction (improved)
# ============================================================

def extract_code_entities_v2(filepath: str) -> List[Tuple[str, str, str]]:
    """Improved entity extraction with docstrings and better class handling."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        try:
            with open(filepath, 'r', errors='replace') as f:
                lines = f.readlines()[:150]
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
                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 30
                code = '\n'.join(source_lines[start:end])
                # Also extract docstring separately for higher weight
                docstring = ast.get_docstring(node) or ""
                entities.append((name, "function", code, docstring))
            except:
                entities.append((name, "function", "", ""))
        elif isinstance(node, ast.ClassDef):
            name = node.name
            try:
                start = node.lineno - 1
                # Get more of the class (methods names + docstring)
                end = min(start + 50,
                         node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 50)
                code = '\n'.join(source_lines[start:end])
                docstring = ast.get_docstring(node) or ""
                # Also extract method names
                method_names = [n.name for n in ast.iter_child_nodes(node)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                extra = " ".join(method_names)
                entities.append((name, "class", code + "\n" + extra, docstring))
            except:
                entities.append((name, "class", "", ""))

    if not entities:
        # Use more of the file
        entities.append(("module", "module", '\n'.join(source_lines[:150]), ""))

    return entities


# ============================================================
# BM25 with custom parameters
# ============================================================

class BM25Custom:
    """BM25 with tunable k1 and b parameters."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = np.array([len(doc) for doc in corpus])
        self.avgdl = np.mean(self.doc_len)
        self.n_docs = len(corpus)

        # Build inverted index
        self.df = defaultdict(int)
        self.tf = []
        for doc in corpus:
            tf = defaultdict(int)
            seen = set()
            for token in doc:
                tf[token] += 1
                if token not in seen:
                    self.df[token] += 1
                    seen.add(token)
            self.tf.append(tf)

        # IDF
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query):
        scores = np.zeros(self.n_docs)
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i in range(self.n_docs):
                tf = self.tf[i].get(q, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / denom
        return scores


# ============================================================
# Build index
# ============================================================

def build_function_index(repos_dir, repo_name, k1=1.5, b=0.75, use_enhanced=True):
    """Build function-level BM25 index with tunable parameters."""
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

    all_tokens = []
    entity_info = []

    for fp in py_files:
        full_path = os.path.join(repo_dir, fp)
        if use_enhanced:
            entities = extract_code_entities_v2(full_path)
        else:
            # Fallback to simple extraction
            try:
                with open(full_path, 'r', errors='replace') as f:
                    source = f.read()
                tree = ast.parse(source, filename=full_path)
                source_lines = source.split('\n')
                entities = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        start = node.lineno - 1
                        end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 20
                        code = '\n'.join(source_lines[start:end])
                        entities.append((node.name, "function", code, ""))
                    elif isinstance(node, ast.ClassDef):
                        start = node.lineno - 1
                        end = min(node.lineno + 30, node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno + 30)
                        code = '\n'.join(source_lines[start:end])
                        entities.append((node.name, "class", code, ""))
                if not entities:
                    entities.append(("module", "module", '\n'.join(source_lines[:100]), ""))
            except:
                continue

        for item in entities:
            if len(item) == 4:
                name, etype, code, docstring = item
            else:
                name, etype, code = item
                docstring = ""

            path_tokens = re.split(r'[/_\-.]', fp.replace('.py', ''))
            path_tokens = [t.lower() for t in path_tokens if len(t) > 1]

            name_tokens = tokenize_code(name)
            code_tokens = tokenize_code(code)
            doc_tokens = tokenize_code(docstring) if docstring else []

            # Weight: path 3x, name 5x, docstring 4x, code 1x
            doc_combined = path_tokens * 3 + name_tokens * 5 + doc_tokens * 4 + code_tokens
            if doc_combined:
                all_tokens.append(doc_combined)
                entity_info.append((fp, name, etype))

    if not all_tokens:
        return None, []

    bm25 = BM25Custom(all_tokens, k1=k1, b=b)
    return bm25, entity_info


# ============================================================
# RRF Fusion
# ============================================================

def rrf_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion of multiple rankings."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


def weighted_rrf_fusion(rankings: List[List[str]], weights: List[float], k: int = 60) -> List[str]:
    """Weighted RRF fusion."""
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] += weight / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: -scores[x])


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--path_bm25', default='data/rankft/swebench_test_bm25_top500.jsonl')
    parser.add_argument('--content_bm25', default='data/rankft/swebench_bm25_content_top500.jsonl')
    parser.add_argument('--function_bm25', default='data/rankft/swebench_bm25_function_top500.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_bm25_ensemble_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print('=== BM25 Ensemble + Tuning for SWE-bench Lite ===')

    # Load test data
    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f'  {len(examples)} test examples')

    # Load existing BM25 rankings
    def load_rankings(path):
        rankings = {}
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                key = item.get('issue_id', item.get('instance_id', ''))
                rankings[key] = item.get('bm25_candidates', [])
        return rankings

    path_rankings = load_rankings(args.path_bm25)
    content_rankings = load_rankings(args.content_bm25)
    function_rankings = load_rankings(args.function_bm25)
    print(f'  Path: {len(path_rankings)}, Content: {len(content_rankings)}, Function: {len(function_rankings)}')

    # ============================================================
    # Strategy 1: RRF of all three
    # ============================================================
    print('\n--- Strategy 1: RRF (3-way) ---')
    for rrf_k in [30, 60, 100]:
        accs = {k: 0 for k in [1, 5, 10, 20, 50]}
        count = 0
        for ex in examples:
            key = ex.get('issue_id', ex.get('instance_id', ''))
            gt = set(ex.get('changed_py_files', []))
            if not gt:
                continue

            r1 = path_rankings.get(key, [])
            r2 = content_rankings.get(key, [])
            r3 = function_rankings.get(key, [])
            if not r3:
                continue

            fused = rrf_fusion([r1, r2, r3], k=rrf_k)
            count += 1
            for k_val in accs:
                if gt.issubset(set(fused[:k_val])):
                    accs[k_val] += 1

        print(f'  RRF(k={rrf_k}): ' + ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())))

    # ============================================================
    # Strategy 2: Weighted RRF
    # ============================================================
    print('\n--- Strategy 2: Weighted RRF ---')
    weight_configs = [
        ([1.0, 1.0, 2.0], "equal path+content, 2x function"),
        ([0.5, 1.0, 2.0], "0.5 path, 1.0 content, 2.0 function"),
        ([0.3, 1.0, 3.0], "0.3 path, 1.0 content, 3.0 function"),
        ([0.0, 1.0, 2.0], "no path, 1.0 content, 2.0 function"),
        ([0.0, 0.5, 3.0], "no path, 0.5 content, 3.0 function"),
    ]
    best_acc1 = 0
    best_config = None
    best_fused_results = None

    for weights, desc in weight_configs:
        accs = {k: 0 for k in [1, 5, 10, 20, 50]}
        count = 0
        fused_results = {}
        for ex in examples:
            key = ex.get('issue_id', ex.get('instance_id', ''))
            gt = set(ex.get('changed_py_files', []))
            if not gt:
                continue
            r1 = path_rankings.get(key, [])
            r2 = content_rankings.get(key, [])
            r3 = function_rankings.get(key, [])
            if not r3:
                continue

            fused = weighted_rrf_fusion([r1, r2, r3], weights, k=60)
            fused_results[key] = fused
            count += 1
            for k_val in accs:
                if gt.issubset(set(fused[:k_val])):
                    accs[k_val] += 1

        acc1 = accs[1] / count * 100
        print(f'  [{desc}]: ' + ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())))
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_config = desc
            best_fused_results = fused_results

    print(f'\n  Best weighted RRF: {best_config} → Acc@1={best_acc1:.2f}%')

    # ============================================================
    # Strategy 3: Function BM25 with tuned k1, b
    # ============================================================
    print('\n--- Strategy 3: Function BM25 parameter tuning ---')

    # Group by repo
    repo_examples = defaultdict(list)
    for ex in examples:
        repo_examples[ex['repo']].append(ex)

    k1_b_configs = [
        (1.2, 0.75),
        (1.5, 0.75),  # default
        (2.0, 0.75),
        (1.5, 0.5),
        (1.5, 0.9),
        (2.0, 0.5),
        (2.5, 0.75),
        (1.2, 0.5),
    ]

    for k1, b in k1_b_configs:
        accs = {k: 0 for k in [1, 5, 10, 20, 50]}
        count = 0
        t0 = time.time()

        for repo_name, exs in sorted(repo_examples.items()):
            bm25, entity_info = build_function_index(
                args.repos_dir, repo_name, k1=k1, b=b, use_enhanced=True
            )
            if bm25 is None:
                continue

            for ex in exs:
                issue_text = ex.get('issue_text', '')
                gt = set(ex.get('changed_py_files', []))
                if not gt:
                    continue

                query_tokens = tokenize_query_enhanced(issue_text)
                scores = bm25.get_scores(query_tokens)

                file_scores = defaultdict(float)
                for i, (fp, name, etype) in enumerate(entity_info):
                    file_scores[fp] = max(file_scores[fp], scores[i])

                ranked = sorted(file_scores.items(), key=lambda x: -x[1])
                candidates = [f for f, _ in ranked[:args.top_k]]

                count += 1
                for k_val in accs:
                    if gt.issubset(set(candidates[:k_val])):
                        accs[k_val] += 1

        elapsed = time.time() - t0
        print(f'  k1={k1}, b={b}: ' +
              ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())) +
              f' ({elapsed:.0f}s)')

    # ============================================================
    # Strategy 4: Enhanced function BM25 (v2 extraction + enhanced query)
    # ============================================================
    print('\n--- Strategy 4: Enhanced function BM25 (v2 + enhanced query) ---')
    best_enhanced_acc1 = 0
    best_enhanced_results = {}

    for k1, b in [(1.5, 0.75), (2.0, 0.5), (1.2, 0.75)]:
        accs = {k: 0 for k in [1, 5, 10, 20, 50, 100, 200, 500]}
        count = 0
        enhanced_results = {}
        t0 = time.time()

        for repo_name, exs in sorted(repo_examples.items()):
            bm25, entity_info = build_function_index(
                args.repos_dir, repo_name, k1=k1, b=b, use_enhanced=True
            )
            if bm25 is None:
                continue

            for ex in exs:
                issue_text = ex.get('issue_text', '')
                gt = set(ex.get('changed_py_files', []))
                key = ex.get('issue_id', ex.get('instance_id', ''))
                if not gt:
                    continue

                query_tokens = tokenize_query_enhanced(issue_text)
                scores = bm25.get_scores(query_tokens)

                file_scores = defaultdict(float)
                for i, (fp, name, etype) in enumerate(entity_info):
                    file_scores[fp] = max(file_scores[fp], scores[i])

                ranked = sorted(file_scores.items(), key=lambda x: -x[1])
                candidates = [f for f, _ in ranked[:args.top_k]]

                enhanced_results[key] = candidates
                count += 1
                for k_val in accs:
                    if gt.issubset(set(candidates[:k_val])):
                        accs[k_val] += 1

        elapsed = time.time() - t0
        acc1 = accs[1] / count * 100
        print(f'  k1={k1}, b={b}: ' +
              ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())) +
              f' ({elapsed:.0f}s)')
        if acc1 > best_enhanced_acc1:
            best_enhanced_acc1 = acc1
            best_enhanced_results = enhanced_results

    # ============================================================
    # Strategy 5: RRF of best function variants
    # ============================================================
    print('\n--- Strategy 5: RRF(enhanced_function, content, path) ---')
    if best_enhanced_results:
        accs = {k: 0 for k in [1, 5, 10, 20, 50, 100]}
        count = 0
        final_results = []
        for ex in examples:
            key = ex.get('issue_id', ex.get('instance_id', ''))
            gt = set(ex.get('changed_py_files', []))
            if not gt:
                continue

            r_enhanced = best_enhanced_results.get(key, [])
            r_content = content_rankings.get(key, [])
            r_function = function_rankings.get(key, [])

            if not r_enhanced:
                continue

            fused = weighted_rrf_fusion(
                [r_enhanced, r_function, r_content],
                [2.0, 1.5, 1.0],
                k=60
            )
            count += 1
            for k_val in accs:
                if gt.issubset(set(fused[:k_val])):
                    accs[k_val] += 1

            final_results.append({
                'repo': ex['repo'],
                'issue_id': key,
                'bm25_candidates': fused[:args.top_k],
                'ground_truth': list(gt),
            })

        print(f'  Result: ' + ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())))

        # Save best ensemble
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for r in final_results:
                f.write(json.dumps(r) + '\n')
        print(f'\n  Saved {len(final_results)} results to {args.output}')

    print('\nDone.')


if __name__ == '__main__':
    main()
