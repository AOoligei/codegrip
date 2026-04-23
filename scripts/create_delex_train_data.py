#!/usr/bin/env python3
"""Create delexicalized TRAINING data for path-debiased reranker training.

Strategy: for each training example, create a delexicalized copy where path
tokens overlapping with issue text are hashed. The training script will mix
normal and delexicalized examples to force the model to learn non-lexical signals.

Output:
  - data/rankft/grepo_train_delex.jsonl (train data with delex'd paths)
  - data/rankft/grepo_train_bm25_top500_delex.jsonl (candidates with delex'd paths)
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_train.jsonl"
BM25_PATH = BASE_DIR / "data" / "rankft" / "grepo_train_bm25_top500.jsonl"
OUT_TRAIN = BASE_DIR / "data" / "rankft" / "grepo_train_delex.jsonl"
OUT_BM25 = BASE_DIR / "data" / "rankft" / "grepo_train_bm25_top500_delex.jsonl"


def stable_hash(s: str, length: int = 6) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def get_issue_tokens(issue_text: str) -> set:
    tokens = set()
    for word in re.findall(r'[a-zA-Z_]\w{2,}', issue_text.lower()):
        tokens.add(word)
    for p in re.findall(r'[\w./]+\.py', issue_text):
        for part in p.replace('.py', '').split('/'):
            for sub in part.split('_'):
                if len(sub) >= 3:
                    tokens.add(sub.lower())
    return tokens


def delexicalize_path(path: str, issue_tokens: set) -> str:
    parts = path.split('/')
    new_parts = []
    for part in parts:
        if part.endswith('.py'):
            base = part[:-3]
            sub_parts = base.split('_')
            new_sub = []
            for s in sub_parts:
                if s.lower() in issue_tokens and len(s) >= 3:
                    new_sub.append(stable_hash(s))
                else:
                    new_sub.append(s)
            new_parts.append('_'.join(new_sub) + '.py')
        else:
            sub_parts = part.split('_')
            new_sub = []
            for s in sub_parts:
                if s.lower() in issue_tokens and len(s) >= 3:
                    new_sub.append(stable_hash(s))
                else:
                    new_sub.append(s)
            new_parts.append('_'.join(new_sub))
    return '/'.join(new_parts)


def main():
    # Load training data
    train_data = []
    with open(TRAIN_PATH) as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"Loaded {len(train_data)} training examples")

    # Load BM25 candidates
    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            d = json.loads(line)
            bm25_data[(d['repo'], d['issue_id'])] = d
    print(f"Loaded {len(bm25_data)} BM25 candidate sets")

    # Process each example
    train_out = []
    bm25_out = {}
    n_tokens_masked = 0
    n_tokens_total = 0
    n_processed = 0

    for ex in train_data:
        key = (ex['repo'], ex['issue_id'])
        if key not in bm25_data:
            continue

        issue_tokens = get_issue_tokens(ex.get('issue_text', ''))

        # Delexicalize GT files
        new_gt = [delexicalize_path(f, issue_tokens) for f in ex.get('changed_py_files', [])]
        new_changed = [delexicalize_path(f, issue_tokens) for f in ex.get('changed_files', [])]

        new_ex = dict(ex)
        new_ex['changed_py_files'] = new_gt
        new_ex['changed_files'] = new_changed
        train_out.append(new_ex)

        # Delexicalize candidates
        bm25_entry = bm25_data[key]
        cand_key = 'candidates' if 'candidates' in bm25_entry else 'bm25_candidates'
        new_cands = [delexicalize_path(c, issue_tokens) for c in bm25_entry[cand_key]]
        new_gt = [delexicalize_path(g, issue_tokens) for g in bm25_entry.get('ground_truth', [])]
        new_bm25 = dict(bm25_entry)
        new_bm25[cand_key] = new_cands
        if 'ground_truth' in new_bm25:
            new_bm25['ground_truth'] = new_gt
        bm25_out[key] = new_bm25

        # Count masking stats
        for f in ex.get('changed_py_files', []):
            parts = f.replace('.py', '').split('/')
            for p in parts:
                for s in p.split('_'):
                    if len(s) >= 3:
                        n_tokens_total += 1
                        if s.lower() in issue_tokens:
                            n_tokens_masked += 1
        n_processed += 1

    # Write outputs
    with open(OUT_TRAIN, 'w') as f:
        for ex in train_out:
            f.write(json.dumps(ex) + '\n')

    with open(OUT_BM25, 'w') as f:
        for entry in bm25_out.values():
            f.write(json.dumps(entry) + '\n')

    print(f"Processed {n_processed} examples")
    print(f"Written {len(train_out)} train examples to {OUT_TRAIN}")
    print(f"Written {len(bm25_out)} candidate sets to {OUT_BM25}")
    print(f"Token masking rate: {n_tokens_masked}/{n_tokens_total} = {n_tokens_masked/max(n_tokens_total,1)*100:.1f}%")


if __name__ == '__main__':
    main()
