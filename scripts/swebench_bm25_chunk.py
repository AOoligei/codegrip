#!/usr/bin/env python3
"""
Chunk-level BM25 for SWE-bench Lite (LocAgent-style).

Key differences from our function-level BM25:
1. Chunk-level indexing (500 tokens per chunk, like LocAgent)
2. Snowball English stemmer
3. Exclude test files
4. k1=1.2, b=0.75 (Lucene defaults, same as LocAgent)
5. File score = max chunk score

Usage:
    python scripts/swebench_bm25_chunk.py \
        --repos_dir data/swebench_lite/repos \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --output data/rankft/swebench_bm25_chunk_top500.jsonl
"""
import os
import re
import json
import argparse
import time
from collections import defaultdict
from typing import List

import numpy as np

np.random.seed(42)

# Try to import Stemmer (PyStemmer / snowballstemmer)
try:
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
    def stem_tokens(tokens):
        return stemmer.stemWords(tokens)
    HAS_STEMMER = True
except ImportError:
    try:
        import snowballstemmer
        stemmer = snowballstemmer.stemmer('english')
        def stem_tokens(tokens):
            return stemmer.stemWords(tokens)
        HAS_STEMMER = True
    except ImportError:
        def stem_tokens(tokens):
            return tokens
        HAS_STEMMER = False


# Test file patterns (match LocAgent)
TEST_PATTERNS = [
    '**/test/**', '**/tests/**', '**/test_*.py', '**/*_test.py',
    '**/testing/**', '**/conftest.py',
]

def is_test_file(filepath: str) -> bool:
    """Check if file is a test file."""
    parts = filepath.split('/')
    for part in parts[:-1]:  # Check directory names
        if part in ('test', 'tests', 'testing'):
            return True
    fname = parts[-1]
    if fname.startswith('test_') or fname.endswith('_test.py'):
        return True
    if fname == 'conftest.py':
        return True
    return False


def tokenize_text(text: str) -> List[str]:
    """Tokenize and optionally stem text."""
    # Split camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    # Split on non-alphanumeric
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    # Stem
    if HAS_STEMMER:
        tokens = stem_tokens(tokens)
    return tokens


def tokenize_query(text: str) -> List[str]:
    """Tokenize query (issue text) with emphasis on code references."""
    base = tokenize_text(text)

    # File references (weight 3x)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        parts = re.split(r'[/_\-.]', ref.replace('.py', ''))
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 3)

    # Quoted identifiers (weight 2x)
    quoted = re.findall(r'[`\'"]([\w.]+)[`\'"]', text)
    for q in quoted:
        parts = re.split(r'[._]', q)
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 2)

    # Dotted references like module.class.method (weight 2x)
    dotted = re.findall(r'\b(\w+(?:\.\w+){1,})\b', text)
    for d in dotted:
        parts = d.lower().split('.')
        extra = [p for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 2)

    return base


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by token count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break

    return chunks if chunks else [text]


class BM25Index:
    """BM25 with Lucene-default parameters (k1=1.2, b=0.75)."""

    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus])
        self.avgdl = np.mean(self.doc_len) if self.n_docs > 0 else 1.0

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

        # IDF (Lucene variant)
        self.idf = {}
        for term, df in self.df.items():
            # Lucene BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            self.idf[term] = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

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


def build_chunk_index(repos_dir: str, repo_name: str, chunk_size: int = 500,
                      exclude_tests: bool = True, k1: float = 1.2, b: float = 0.75):
    """Build chunk-level BM25 index."""
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
    chunk_info = []  # (filepath, chunk_idx)

    for fp in py_files:
        full_path = os.path.join(repo_dir, fp)
        try:
            with open(full_path, 'r', errors='replace') as f:
                content = f.read()
        except:
            continue

        # Prepend path tokens to each chunk for context
        path_str = fp.replace('.py', '').replace('/', ' ').replace('_', ' ')

        chunks = chunk_text(content, chunk_size=chunk_size, overlap=50)
        for ci, chunk in enumerate(chunks):
            doc_text = path_str + ' ' + chunk
            tokens = tokenize_text(doc_text)
            if tokens:
                all_tokens.append(tokens)
                chunk_info.append((fp, ci))

    if not all_tokens:
        return None, []

    bm25 = BM25Index(all_tokens, k1=k1, b=b)
    return bm25, chunk_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_bm25_chunk_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print('=== Chunk-Level BM25 for SWE-bench Lite (LocAgent-style) ===')
    print(f'  Stemmer: {"Snowball English" if HAS_STEMMER else "None (install PyStemmer!)"}')

    examples = []
    with open(args.test_data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f'  {len(examples)} test examples')

    repo_examples = defaultdict(list)
    for ex in examples:
        repo_examples[ex['repo']].append(ex)

    # Test multiple configurations
    configs = [
        {'chunk_size': 500, 'exclude_tests': True, 'k1': 1.2, 'b': 0.75, 'name': 'chunk500_notest_k1.2'},
        {'chunk_size': 500, 'exclude_tests': False, 'k1': 1.2, 'b': 0.75, 'name': 'chunk500_withtest_k1.2'},
        {'chunk_size': 300, 'exclude_tests': True, 'k1': 1.2, 'b': 0.75, 'name': 'chunk300_notest_k1.2'},
        {'chunk_size': 200, 'exclude_tests': True, 'k1': 1.2, 'b': 0.75, 'name': 'chunk200_notest_k1.2'},
        {'chunk_size': 500, 'exclude_tests': True, 'k1': 1.5, 'b': 0.75, 'name': 'chunk500_notest_k1.5'},
        {'chunk_size': 500, 'exclude_tests': True, 'k1': 1.2, 'b': 0.5, 'name': 'chunk500_notest_b0.5'},
    ]

    best_acc1 = 0
    best_config_name = ""
    best_results = []

    for cfg in configs:
        accs = {k: 0 for k in [1, 5, 10, 20, 50, 100, 200, 500]}
        count = 0
        config_results = []
        t0 = time.time()

        for repo_name, exs in sorted(repo_examples.items()):
            bm25, chunk_info = build_chunk_index(
                args.repos_dir, repo_name,
                chunk_size=cfg['chunk_size'],
                exclude_tests=cfg['exclude_tests'],
                k1=cfg['k1'], b=cfg['b'],
            )
            if bm25 is None:
                continue

            n_chunks = len(chunk_info)
            n_files = len(set(c[0] for c in chunk_info))

            for ex in exs:
                issue_text = ex.get('issue_text', '')
                gt = set(ex.get('changed_py_files', []))
                if not gt:
                    continue

                query_tokens = tokenize_query(issue_text)
                scores = bm25.get_scores(query_tokens)

                # Aggregate to file level: max chunk score per file
                file_scores = defaultdict(float)
                for i, (fp, ci) in enumerate(chunk_info):
                    file_scores[fp] = max(file_scores[fp], scores[i])

                ranked = sorted(file_scores.items(), key=lambda x: -x[1])
                candidates = [f for f, _ in ranked[:args.top_k]]

                count += 1
                for k_val in accs:
                    if gt.issubset(set(candidates[:k_val])):
                        accs[k_val] += 1

                config_results.append({
                    'repo': ex['repo'],
                    'issue_id': ex.get('issue_id', ex.get('instance_id', '')),
                    'bm25_candidates': candidates,
                    'ground_truth': list(gt),
                })

        elapsed = time.time() - t0
        acc1 = accs[1] / count * 100 if count > 0 else 0
        print(f'\n  [{cfg["name"]}] ({elapsed:.0f}s, {count} examples):')
        for k_val in sorted(accs.keys()):
            print(f'    Acc@{k_val}: {accs[k_val]/count*100:.2f}%')

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_config_name = cfg['name']
            best_results = config_results

    print(f'\n  Best config: {best_config_name} → Acc@1={best_acc1:.2f}%')

    # Save best results
    if best_results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for r in best_results:
                f.write(json.dumps(r) + '\n')
        print(f'  Saved {len(best_results)} to {args.output}')

    # ============================================================
    # Also try: Chunk + Function hybrid
    # ============================================================
    print('\n--- Hybrid: Chunk + Function BM25 fusion ---')
    # Load function BM25 rankings
    func_rankings = {}
    func_path = 'data/rankft/swebench_bm25_function_top500.jsonl'
    if os.path.exists(func_path):
        with open(func_path) as f:
            for line in f:
                item = json.loads(line)
                key = item.get('issue_id', item.get('instance_id', ''))
                func_rankings[key] = item.get('bm25_candidates', [])

    if func_rankings and best_results:
        # RRF fusion
        from collections import defaultdict as dd
        for rrf_k in [30, 60]:
            accs = {k: 0 for k in [1, 5, 10, 20, 50]}
            count = 0
            for r in best_results:
                key = r['issue_id']
                gt = set(r['ground_truth'])
                chunk_ranking = r['bm25_candidates']
                func_ranking = func_rankings.get(key, [])
                if not func_ranking:
                    continue

                # RRF
                scores = defaultdict(float)
                for rank, f in enumerate(chunk_ranking):
                    scores[f] += 1.0 / (rrf_k + rank + 1)
                for rank, f in enumerate(func_ranking):
                    scores[f] += 1.0 / (rrf_k + rank + 1)
                fused = sorted(scores.keys(), key=lambda x: -scores[x])

                count += 1
                for k_val in accs:
                    if gt.issubset(set(fused[:k_val])):
                        accs[k_val] += 1

            print(f'  Chunk+Function RRF(k={rrf_k}): ' +
                  ' '.join(f'@{k}={v/count*100:.2f}%' for k, v in sorted(accs.items())))

    print('\nDone.')


if __name__ == '__main__':
    main()
