#!/usr/bin/env python3
"""Create random-masked path perturbation data (control for delexicalization).

For each test example, mask the same NUMBER of path tokens that
delexicalization would mask, but choose them RANDOMLY (not based on
issue-text overlap). This controls for "just augmentation" — if
random masking causes similar collapse as delex, then the effect is
generic noise sensitivity, not shortcut removal.
"""
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "data" / "grepo_text" / "grepo_test.jsonl"
BM25_PATH = BASE_DIR / "data" / "rankft" / "merged_bm25_exp6_candidates.jsonl"
OUT_DIR = BASE_DIR / "experiments" / "path_perturb_random_mask"


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


def count_overlap_tokens(path: str, issue_tokens: set) -> int:
    """Count how many path sub-tokens overlap with issue tokens."""
    parts = path.split('/')
    count = 0
    for part in parts:
        base = part[:-3] if part.endswith('.py') else part
        for s in base.split('_'):
            if len(s) >= 3 and s.lower() in issue_tokens:
                count += 1
    return count


def get_all_maskable_positions(path: str):
    """Return list of (part_idx, sub_idx, sub_token) for all maskable tokens."""
    parts = path.split('/')
    positions = []
    for pi, part in enumerate(parts):
        base = part[:-3] if part.endswith('.py') else part
        for si, s in enumerate(base.split('_')):
            if len(s) >= 3:
                positions.append((pi, si, s))
    return positions


def random_mask_path(path: str, n_to_mask: int, rng: random.Random) -> str:
    """Mask n_to_mask randomly chosen path tokens (len >= 3)."""
    parts = path.split('/')
    positions = get_all_maskable_positions(path)

    if n_to_mask <= 0 or not positions:
        return path

    n_to_mask = min(n_to_mask, len(positions))
    chosen = set(rng.sample(range(len(positions)), n_to_mask))

    # Build lookup: (part_idx, sub_idx) -> should_mask
    mask_set = set()
    for idx in chosen:
        pi, si, _ = positions[idx]
        mask_set.add((pi, si))

    # Reconstruct path
    new_parts = []
    for pi, part in enumerate(parts):
        has_py = part.endswith('.py')
        base = part[:-3] if has_py else part
        subs = base.split('_')
        new_subs = []
        for si, s in enumerate(subs):
            if (pi, si) in mask_set:
                new_subs.append(stable_hash(s))
            else:
                new_subs.append(s)
        new_part = '_'.join(new_subs)
        if has_py:
            new_part += '.py'
        new_parts.append(new_part)
    return '/'.join(new_parts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(42)

    # Load data
    test_data = []
    with open(TEST_PATH) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test examples")

    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            d = json.loads(line)
            bm25_data[(d['repo'], d['issue_id'])] = d
    print(f"Loaded {len(bm25_data)} BM25 candidate sets")

    # Process
    test_out = []
    bm25_out = []
    n_masked_total = 0
    n_maskable_total = 0
    n_overlap_total = 0

    for ex in test_data:
        key = (ex['repo'], ex['issue_id'])
        if key not in bm25_data:
            continue

        issue_tokens = get_issue_tokens(ex.get('issue_text', ''))
        bm25_entry = bm25_data[key]

        # For each candidate, count how many tokens delex would mask,
        # then mask that many randomly instead
        all_paths = (
            ex.get('changed_py_files', []) +
            ex.get('changed_files', []) +
            bm25_entry['candidates']
        )

        # Build per-path mask counts based on issue overlap
        path_mask_counts = {}
        for p in set(all_paths):
            n_overlap = count_overlap_tokens(p, issue_tokens)
            n_maskable = len(get_all_maskable_positions(p))
            path_mask_counts[p] = n_overlap
            n_overlap_total += n_overlap
            n_maskable_total += n_maskable

        # Random-mask GT files
        new_gt = []
        for f in ex.get('changed_py_files', []):
            n = path_mask_counts.get(f, 0)
            new_gt.append(random_mask_path(f, n, rng))
            n_masked_total += n

        new_changed = []
        for f in ex.get('changed_files', []):
            n = path_mask_counts.get(f, 0)
            new_changed.append(random_mask_path(f, n, rng))

        new_ex = dict(ex)
        new_ex['changed_py_files'] = new_gt
        new_ex['changed_files'] = new_changed
        test_out.append(new_ex)

        # Random-mask candidates
        new_cands = []
        for c in bm25_entry['candidates']:
            n = path_mask_counts.get(c, 0)
            new_cands.append(random_mask_path(c, n, rng))

        bm25_out.append({
            'repo': bm25_entry['repo'],
            'issue_id': bm25_entry['issue_id'],
            'candidates': new_cands
        })

    # Write
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
    print(f"Overlap tokens (delex would mask): {n_overlap_total}")
    print(f"Maskable tokens (total): {n_maskable_total}")
    print(f"Overlap rate: {n_overlap_total/max(n_maskable_total,1)*100:.1f}%")
    print(f"Random-masked tokens: {n_masked_total}")


if __name__ == '__main__':
    main()
