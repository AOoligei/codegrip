#!/usr/bin/env python3
"""
Prepare SWE-bench Full training data for reranker fine-tuning.

Uses SWE-bench Full (2294) minus Lite (300) = 1994 training examples.
Extracts changed files from patches, builds BM25 candidates using
our existing chunk-level BM25 infrastructure.

Output format matches GREPO training data for train_rankft.py compatibility.
"""
import os
import json
import re
import argparse
import time
from collections import defaultdict
from typing import List, Set

import numpy as np
np.random.seed(42)

from datasets import load_dataset

# Import BM25 infrastructure from our chunk BM25 script
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


def tokenize_query(text: str) -> List[str]:
    base = tokenize_text(text)
    file_refs = re.findall(r'[\w/]+\.py\b', text)
    for ref in file_refs:
        parts = re.split(r'[/_.\-]', ref.replace('.py', ''))
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 3)
    quoted = re.findall(r'[`\'"]([\w.]+)[`\'"]', text)
    for q in quoted:
        parts = re.split(r'[._]', q)
        extra = [p.lower() for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 2)
    dotted = re.findall(r'\b(\w+(?:\.\w+){1,})\b', text)
    for d in dotted:
        parts = d.lower().split('.')
        extra = [p for p in parts if len(p) > 1]
        if HAS_STEMMER:
            extra = stem_tokens(extra)
        base.extend(extra * 2)
    return base


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


class BM25Index:
    def __init__(self, corpus, k1=1.5, b=0.75):
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


def extract_changed_files(patch: str) -> List[str]:
    """Extract changed .py file paths from a git diff patch."""
    files = re.findall(r'diff --git a/(.+?) b/', patch)
    py_files = [f for f in files if f.endswith('.py')]
    return py_files


def build_repo_index(repos_dir: str, repo_name: str):
    """Build chunk-level BM25 index for a repo (including test files for training)."""
    repo_dir = os.path.join(repos_dir, repo_name)
    if not os.path.isdir(repo_dir):
        return None, [], set()

    py_files = []
    all_files = set()
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, repo_dir)
                py_files.append(rel_path)
                all_files.add(rel_path)

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
        return None, [], all_files

    bm25 = BM25Index(all_tokens, k1=1.5, b=0.75)
    return bm25, chunk_info, all_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos_dir', default='data/swebench_lite/repos')
    parser.add_argument('--output_train', default='data/swebench_train/swebench_train.jsonl')
    parser.add_argument('--output_bm25', default='data/swebench_train/swebench_train_bm25_top500.jsonl')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    print("=== Preparing SWE-bench Full Training Data ===")
    print(f"  Stemmer: {'Snowball English' if HAS_STEMMER else 'None'}")

    # Load SWE-bench Full and Lite
    print("  Loading datasets from HuggingFace...")
    full = load_dataset('princeton-nlp/SWE-bench', split='test')
    lite = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    lite_ids = set(lite['instance_id'])

    train_examples = [ex for ex in full if ex['instance_id'] not in lite_ids]
    print(f"  Training pool: {len(train_examples)} examples (Full - Lite)")

    # Group by repo
    repo_examples = defaultdict(list)
    for ex in train_examples:
        repo_dir_name = ex['repo'].replace('/', '__')
        repo_examples[repo_dir_name].append(ex)

    # Process each repo
    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_bm25), exist_ok=True)

    train_data = []
    bm25_data = []
    skipped = 0
    no_gt_in_repo = 0

    t0 = time.time()
    for repo_name, exs in sorted(repo_examples.items()):
        print(f"\n  Processing {repo_name} ({len(exs)} examples)...")
        bm25, chunk_info, all_files = build_repo_index(args.repos_dir, repo_name)
        if bm25 is None:
            print(f"    Skipped (no index)")
            skipped += len(exs)
            continue

        n_files = len(all_files)
        n_chunks = len(chunk_info)
        print(f"    {n_files} files, {n_chunks} chunks")

        for ex in exs:
            # Extract GT files from patch
            changed_files = extract_changed_files(ex['patch'])
            if not changed_files:
                skipped += 1
                continue

            # Check if GT files exist in repo
            existing_gt = [f for f in changed_files if f in all_files]
            if not existing_gt:
                no_gt_in_repo += 1
                continue

            issue_text = ex.get('problem_statement', '')
            instance_id = ex['instance_id']

            # Build BM25 candidates
            query_tokens = tokenize_query(issue_text)
            scores = bm25.get_scores(query_tokens)

            # Aggregate to file level
            file_scores = defaultdict(float)
            for i, (fp, ci) in enumerate(chunk_info):
                file_scores[fp] = max(file_scores[fp], scores[i])

            ranked = sorted(file_scores.items(), key=lambda x: -x[1])
            candidates = [f for f, _ in ranked[:args.top_k]]

            # Training data format (GREPO-compatible)
            train_item = {
                'repo': ex['repo'].replace('/', '__'),
                'issue_id': instance_id,
                'issue_text': issue_text,
                'changed_py_files': changed_files,
                'changed_files': changed_files,
            }
            train_data.append(train_item)

            # BM25 candidates format
            bm25_item = {
                'repo': ex['repo'].replace('/', '__'),
                'issue_id': instance_id,
                'ground_truth': changed_files,
                'bm25_candidates': candidates,
            }
            bm25_data.append(bm25_item)

        elapsed = time.time() - t0
        print(f"    Done. Total so far: {len(train_data)} examples ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    print(f"  Total training examples: {len(train_data)}")
    print(f"  Skipped (no patch/index): {skipped}")
    print(f"  Skipped (GT not in repo): {no_gt_in_repo}")

    # Compute BM25 recall stats
    accs = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    for item in bm25_data:
        gt = set(item['ground_truth'])
        cands = item['bm25_candidates']
        for k in accs:
            if gt.issubset(set(cands[:k])):
                accs[k] += 1
    n = len(bm25_data)
    print(f"\n  BM25 recall on training data ({n} examples):")
    for k in sorted(accs.keys()):
        print(f"    R@{k}: {accs[k]/n*100:.2f}%")

    # Save
    with open(args.output_train, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"\n  Saved training data: {args.output_train}")

    with open(args.output_bm25, 'w') as f:
        for item in bm25_data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved BM25 candidates: {args.output_bm25}")


if __name__ == '__main__':
    main()
