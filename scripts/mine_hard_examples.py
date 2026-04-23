#!/usr/bin/env python3
"""
Mine hard examples for SPECTER code expert training.

Hard = examples where path-only model is uncertain or wrong. These are
where code reasoning could potentially help.

Mining criteria (train on positive examples from training set):
1. Path-confusable: GT shares same directory or same filename stem with
   at least one BM25 candidate (path can't easily distinguish)
2. Low path-overlap: Jaccard(issue_tokens, gt_path_tokens) < 0.2
3. High path score std: path-only gives similar scores to multiple candidates
   (proxy: GT has many same-stem files in candidate pool)

For each mined example, we also generate:
- Hard negatives: same-directory and same-stem files (force model to use code)
- Easy negatives: random files (for balance)

Output:
- hard_examples_train.jsonl: (repo, issue_id, gt_file, hard_negs, easy_negs, features)

Usage:
    python scripts/mine_hard_examples.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/specter/data
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

np.random.seed(42)

TRAIN_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_train.jsonl"
BM25_TRAIN_PATH = "/home/chenlibin/grepo_agent/data/rankft/grepo_train_bm25_top500.jsonl"
TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_TEST_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"


def tokenize_path(path):
    """Tokenize path for overlap computation."""
    return set(t.lower() for t in re.split(r'[/._\-]', path) if t)


def path_coverage_by_issue(issue_text, path):
    """Fraction of path tokens that appear in the issue text.

    This is NOT Jaccard — it measures coverage (path-side recall), which
    is more meaningful here since path tokens (~5) vs issue tokens (~100)
    makes true Jaccard dominated by the issue size.

    Tokenize both on same separators so "foo_bar" in issue matches
    "foo_bar.py" components.
    """
    issue_tokens = set(t for t in re.split(r'[/._\-\s,;:!?()\[\]{}"\'`<>]+',
                                            issue_text.lower()) if t)
    path_tokens = tokenize_path(path)
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def stem_normalize(path):
    """Normalize filename stem: strip test_/test prefixes/suffixes."""
    stem = os.path.splitext(os.path.basename(path))[0]
    # Remove test prefixes/suffixes
    stem = re.sub(r'^test_', '', stem)
    stem = re.sub(r'_test$', '', stem)
    stem = re.sub(r'^tests_', '', stem)
    stem = re.sub(r'_tests$', '', stem)
    return stem


def find_path_confusable_negatives(gt_file, all_gt_files, candidates, max_neg=8):
    """Find candidates that are path-similar to GT but not in any GT file set.

    all_gt_files: set of all GT files for this issue (to avoid false negatives
    on multi-file bugs where another GT file might be same-dir/same-stem).

    Priority:
    1. Same directory + same normalized stem (strongest confuser)
    2. Same directory, different stem
    3. Different directory, same normalized stem
    """
    gt_dir = os.path.dirname(gt_file)
    gt_stem = stem_normalize(gt_file)
    gt_set = set(all_gt_files)

    same_dir_same_stem = []
    same_dir = []
    same_stem = []

    for c in candidates:
        if c in gt_set:
            continue
        c_dir = os.path.dirname(c)
        c_stem = stem_normalize(c)

        if c_dir == gt_dir and c_stem == gt_stem:
            same_dir_same_stem.append(c)
        elif c_dir == gt_dir:
            same_dir.append(c)
        elif c_stem == gt_stem:
            same_stem.append(c)

    hard_negs = same_dir_same_stem + same_dir + same_stem
    return hard_negs[:max_neg]


def mine_training_examples(args):
    """Mine hard training examples."""
    print("Loading training data...")
    train_data = []
    with open(TRAIN_PATH) as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"  {len(train_data)} training examples")

    bm25_data = {}
    with open(BM25_TRAIN_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec
    print(f"  {len(bm25_data)} BM25 candidates")

    hard_examples = []
    stats = defaultdict(int)

    for rec in train_data:
        repo = rec["repo"]
        issue_id = str(rec["issue_id"])
        issue_text = rec["issue_text"]
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))

        if not gt_files:
            continue

        key = (repo, issue_id)
        if key not in bm25_data:
            continue

        candidates = bm25_data[key].get("candidates",
                                         bm25_data[key].get("bm25_candidates", []))
        if not candidates:
            continue

        # For each GT file, check if it's hard
        for gt_file in gt_files:
            if gt_file not in candidates:
                stats["gt_not_in_candidates"] += 1
                continue

            # Feature 1: path coverage by issue (fraction of path tokens in issue)
            p_overlap = path_coverage_by_issue(issue_text, gt_file)

            # Feature 2: find path-confusable negatives (excluding all GT files)
            hard_negs = find_path_confusable_negatives(
                gt_file, gt_files, candidates[:50],
                max_neg=args.max_hard_negs)

            # Criteria for "hard": BOTH low overlap AND has at least 1 confusable negative
            # (low overlap alone → no signal for code; confusable alone → may still be path-obvious)
            is_low_overlap = p_overlap < args.overlap_threshold
            is_confusable = len(hard_negs) >= 1

            if not (is_low_overlap and is_confusable):
                stats["not_hard"] += 1
                continue

            # Sample easy negatives (random from non-GT candidates)
            non_gt = [c for c in candidates[:100]
                      if c not in gt_files and c not in hard_negs]
            easy_negs = list(np.random.choice(
                non_gt, size=min(args.max_easy_negs, len(non_gt)),
                replace=False)) if non_gt else []

            hard_examples.append({
                "repo": repo,
                "issue_id": issue_id,
                "issue_text": issue_text,
                "gt_file": gt_file,
                "hard_negs": hard_negs,
                "easy_negs": easy_negs,
                "path_overlap": p_overlap,
                "is_low_overlap": is_low_overlap,
                "is_confusable": is_confusable,
                "num_hard_negs": len(hard_negs),
            })
            stats["hard"] += 1

    print(f"\n=== Mining stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\n=== Hard examples: {len(hard_examples)} ===")
    print(f"  Low overlap: {sum(1 for e in hard_examples if e['is_low_overlap'])}")
    print(f"  Confusable: {sum(1 for e in hard_examples if e['is_confusable'])}")
    print(f"  Both: {sum(1 for e in hard_examples if e['is_low_overlap'] and e['is_confusable'])}")
    if hard_examples:
        print(f"  Avg hard negs: {np.mean([e['num_hard_negs'] for e in hard_examples]):.1f}")
    else:
        print("  WARNING: no hard examples mined — check thresholds")

    return hard_examples


def mine_test_examples(args):
    """Mine hard test examples for evaluation slices."""
    print("\nLoading test data...")
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    bm25_data = {}
    with open(BM25_TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec

    hard_test_keys = []
    for key, rec in test_data.items():
        if key not in bm25_data:
            continue
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        if not gt_files:
            continue

        candidates = bm25_data[key].get("candidates",
                                         bm25_data[key].get("bm25_candidates", []))

        # Check if any GT file is in a hard condition (same criteria as training)
        is_hard = False
        for gt in gt_files:
            if gt not in candidates:
                continue
            p_overlap = path_coverage_by_issue(rec["issue_text"], gt)
            hard_negs = find_path_confusable_negatives(
                gt, gt_files, candidates[:50],
                max_neg=args.max_hard_negs)
            if p_overlap < args.overlap_threshold and len(hard_negs) >= 1:
                is_hard = True
                break

        if is_hard:
            hard_test_keys.append({
                "repo": rec["repo"],
                "issue_id": str(rec["issue_id"]),
            })

    print(f"  Hard test examples: {len(hard_test_keys)} / {len(test_data)}")
    return hard_test_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.20)
    parser.add_argument("--max_hard_negs", type=int, default=8)
    parser.add_argument("--max_easy_negs", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Mine training examples
    hard_train = mine_training_examples(args)

    # Mine test slice
    hard_test = mine_test_examples(args)

    # Save
    train_path = os.path.join(args.output_dir, "hard_examples_train.jsonl")
    with open(train_path, "w") as f:
        for ex in hard_train:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved training: {train_path}")

    test_path = os.path.join(args.output_dir, "hard_test_keys.jsonl")
    with open(test_path, "w") as f:
        for ex in hard_test:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved test slice: {test_path}")

    summary = {
        "num_hard_train": len(hard_train),
        "num_hard_test": len(hard_test),
        "overlap_threshold": args.overlap_threshold,
        "max_hard_negs": args.max_hard_negs,
        "max_easy_negs": args.max_easy_negs,
    }
    with open(os.path.join(args.output_dir, "mining_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
