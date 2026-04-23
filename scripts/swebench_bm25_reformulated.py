#!/usr/bin/env python3
"""
BM25 retrieval using LLM-reformulated queries for SWE-bench Lite.

After running swebench_llm_reformulate.py to get keywords,
this script uses those keywords as BM25 queries (chunk-level + function-level)
with all our best tricks: test exclusion, stemming, RRF fusion.
"""
import os
import json
import re
import argparse
from collections import defaultdict
from typing import List

import numpy as np

np.random.seed(42)

# Reuse our BM25 infrastructure
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


def tokenize_text(text: str) -> List[str]:
    text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\\1 \\2', text)
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    tokens = [t for t in tokens if len(t) > 1]
    if HAS_STEMMER:
        tokens = stem_tokens(tokens)
    return tokens


def tokenize_keywords(keywords: List[str], issue_text: str) -> List[str]:
    """Tokenize LLM-extracted keywords with boosting."""
    tokens = []

    # Keyword tokens (high weight — these are LLM-curated)
    for kw in keywords:
        kw_tokens = tokenize_text(kw)
        tokens.extend(kw_tokens * 3)  # 3x weight

    # File path references from keywords (extra boost)
    for kw in keywords:
        if '/' in kw or kw.endswith('.py'):
            parts = re.split(r'[/_.\-]', kw.replace('.py', ''))
            extra = [p.lower() for p in parts if len(p) > 1]
            if HAS_STEMMER:
                extra = stem_tokens(extra)
            tokens.extend(extra * 5)  # 5x for explicit file paths

    # Also include some issue text tokens (lower weight, 1x)
    issue_tokens = tokenize_text(issue_text[:1000])
    tokens.extend(issue_tokens)

    return tokens


class BM25Index:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus])
        self.avgdl = np.mean(self.doc_len) if self.n_docs > 0 else 1.0
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
        self.idf = {}
        for term, df in self.df.items():
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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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


def build_chunk_index(repos_dir, repo_name, k1=1.5, b=0.75):
    """Build chunk-level BM25 index (test excluded, stemmed)."""
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
                if not is_test_file(rel_path):
                    py_files.append(rel_path)

    all_tokens = []
    chunk_info = []
    for fp in py_files:
        full_path = os.path.join(repo_dir, fp)
        try:
            with open(full_path, 'r', errors='replace') as f:
                content = f.read()
        except:
            continue
        path_str = fp.replace('.py', '').replace('/', ' ').replace('_', ' ')
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        for ci, chunk in enumerate(chunks):
            doc_text = path_str + ' ' + chunk
            tokens = tokenize_text(doc_text)
            if tokens:
                all_tokens.append(tokens)
                chunk_info.append((fp, ci))

    if not all_tokens:
        return None, []
    return BM25Index(all_tokens, k1=k1, b=b), chunk_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--test_data', default='data/swebench_lite/swebench_lite_test.jsonl')
    parser.add_argument('--keywords_file', default='data/rankft/swebench_llm_keywords.jsonl')
    parser.add_argument('--output', default='data/rankft/swebench_bm25_reformulated_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print('=== BM25 with LLM-Reformulated Queries ===')
    print(f'  Stemmer: {"Snowball English" if HAS_STEMMER else "None"}')

    # Load test data
    examples = {}
    with open(args.test_data) as f:
        for line in f:
            d = json.loads(line)
            key = d.get('issue_id', d.get('instance_id', ''))
            examples[key] = d

    # Load LLM keywords
    keywords_map = {}
    with open(args.keywords_file) as f:
        for line in f:
            d = json.loads(line)
            keywords_map[d['issue_id']] = d['keywords']
    print(f'  {len(keywords_map)} keyword sets loaded')

    # Group by repo
    repo_examples = defaultdict(list)
    for key, ex in examples.items():
        repo_examples[ex['repo']].append((key, ex))

    # Process each repo
    results = []
    accs = {k: 0 for k in [1, 5, 10, 20, 50, 100]}
    count = 0

    for repo_name, exs in sorted(repo_examples.items()):
        bm25, chunk_info = build_chunk_index(args.repos_dir, repo_name)
        if bm25 is None:
            print(f'  Skipped {repo_name} (no index)')
            continue

        for key, ex in exs:
            issue_text = ex.get('issue_text', ex.get('problem_statement', ''))
            gt = set(ex.get('changed_py_files', []))
            if not gt:
                continue

            # Use LLM keywords if available, otherwise fall back to issue text
            keywords = keywords_map.get(key, [])
            query_tokens = tokenize_keywords(keywords, issue_text)

            scores = bm25.get_scores(query_tokens)

            # Aggregate to file level
            file_scores = defaultdict(float)
            for i, (fp, ci) in enumerate(chunk_info):
                file_scores[fp] = max(file_scores[fp], scores[i])

            ranked = sorted(file_scores.items(), key=lambda x: -x[1])
            candidates = [f for f, _ in ranked[:args.top_k]]

            count += 1
            for k_val in accs:
                if gt.issubset(set(candidates[:k_val])):
                    accs[k_val] += 1

            results.append({
                'repo': ex['repo'],
                'issue_id': key,
                'bm25_candidates': candidates,
                'ground_truth': list(gt),
            })

    print(f'\n  Results ({count} examples):')
    for k_val in sorted(accs.keys()):
        print(f'    Acc@{k_val}: {accs[k_val]/count*100:.2f}%')

    # Save
    if results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f'  Saved {len(results)} to {args.output}')


if __name__ == '__main__':
    main()
