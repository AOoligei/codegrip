#!/usr/bin/env python3
"""Create delexicalized path perturbation data.

For each test example, hash any path tokens that literally appear in the
issue text. This isolates the lexical shortcut channel.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

random_seed = 42

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
BM25_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"
OUT_DIR = BASE_DIR / "experiments" / "path_perturb_delexicalize"


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
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load test data
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test examples")

    # Load BM25 candidates indexed by (repo, issue_id)
    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            d = json.loads(line)
            bm25_data[(d['repo'], d['issue_id'])] = d
    print(f"Loaded {len(bm25_data)} BM25 candidate sets")

    # Process each example
    test_out = []
    bm25_out = []
    n_tokens_masked = 0
    n_tokens_total = 0

    for ex in test_data:
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
        test_out.append(new_ex)

        # Delexicalize candidates
        bm25_entry = bm25_data[key]
        new_cands = [delexicalize_path(c, issue_tokens) for c in bm25_entry['candidates']]
        bm25_out.append({
            'repo': bm25_entry['repo'],
            'issue_id': bm25_entry['issue_id'],
            'candidates': new_cands
        })

        # Count masking stats
        for f in ex.get('changed_py_files', []):
            parts = f.replace('.py', '').split('/')
            for p in parts:
                for s in p.split('_'):
                    if len(s) >= 3:
                        n_tokens_total += 1
                        if s.lower() in issue_tokens:
                            n_tokens_masked += 1

    # Write outputs
    test_out_path = OUT_DIR / "test.jsonl"
    bm25_out_path = OUT_DIR / "bm25_candidates.jsonl"

    with open(test_out_path, 'w') as f:
        for ex in test_out:
            f.write(json.dumps(ex) + '\n')

    with open(bm25_out_path, 'w') as f:
        for ex in bm25_out:
            f.write(json.dumps(ex) + '\n')

    print(f"Written {len(test_out)} test examples to {test_out_path}")
    print(f"Written {len(bm25_out)} candidate sets to {bm25_out_path}")
    print(f"Token masking rate: {n_tokens_masked}/{n_tokens_total} = {n_tokens_masked/max(n_tokens_total,1)*100:.1f}%")


if __name__ == '__main__':
    main()
