"""
Report R@1 on multiple diagnostic slices for a given eval result.

Usage:
    python scripts/eval_slice_analysis.py \
        --predictions <dir>/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl

Deterministic (seed 42), CPU-only, standard libraries + numpy.
"""

import argparse
import json
import re
import numpy as np


def load_jsonl(path):
    """Load JSONL, return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_predictions(path):
    """Load predictions, return dict keyed by (repo, issue_id_str)."""
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            data[key] = rec
    return data


def make_key(rec):
    """Make a (repo, issue_id_str) key from any record."""
    return (rec["repo"], str(rec["issue_id"]))


def compute_hit_at_1(rec):
    """Compute hit@1."""
    gt = set(rec["ground_truth"])
    predicted = rec["predicted"]
    if len(predicted) == 0:
        return 0.0
    return 1.0 if predicted[0] in gt else 0.0


def tokenize_path(path):
    """Tokenize a file path into tokens."""
    parts = re.split(r'[/._\-]', path.lower())
    return set(p for p in parts if p)


def compute_issue_path_jaccard(issue_text, changed_files):
    """Compute Jaccard between issue text tokens and path tokens."""
    issue_tokens = set(re.findall(r'[a-zA-Z0-9]+', issue_text.lower()))
    path_tokens = set()
    for f in changed_files:
        path_tokens |= tokenize_path(f)
    if not issue_tokens and not path_tokens:
        return 0.0
    intersection = len(issue_tokens & path_tokens)
    union = len(issue_tokens | path_tokens)
    return intersection / union if union > 0 else 0.0


def compute_slice_r1(predictions, key_set):
    """Compute R@1 for a subset of keys."""
    hits = []
    for k in key_set:
        if k in predictions:
            hits.append(compute_hit_at_1(predictions[k]))
    if not hits:
        return 0.0, 0
    return np.mean(hits), len(hits)


def main():
    parser = argparse.ArgumentParser(description="Eval slice analysis")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test_data", default="data/grepo_text/grepo_test.jsonl")
    args = parser.parse_args()

    np.random.seed(42)

    # Load data
    predictions = load_predictions(args.predictions)
    test_data = load_jsonl(args.test_data)

    # Build key sets for each slice
    all_test_keys = set(make_key(r) for r in test_data)
    pred_keys = set(predictions.keys())

    # Slice 1: Full test (intersection of test and predictions)
    full_keys = all_test_keys & pred_keys

    # Slice 2: Code-Crucial v2 strict
    strict_data = load_jsonl("data/code_crucial_v2_strict.jsonl")
    strict_keys = set(make_key(r) for r in strict_data) & pred_keys

    # Slice 3: Code-Crucial v2 broad
    broad_data = load_jsonl("data/code_crucial_v2_broad.jsonl")
    broad_keys_all = {make_key(r): r for r in broad_data}
    broad_keys = set(broad_keys_all.keys()) & pred_keys

    # Slice 4: Low Jaccard (bottom quartile by issue-path overlap)
    jaccards = {}
    for rec in test_data:
        key = make_key(rec)
        if key in pred_keys:
            jaccards[key] = compute_issue_path_jaccard(rec["issue_text"], rec["changed_files"])

    if jaccards:
        jacc_values = np.array(list(jaccards.values()))
        q25 = np.percentile(jacc_values, 25)
        low_jacc_keys = set(k for k, v in jaccards.items() if v <= q25)
    else:
        low_jacc_keys = set()

    # Slice 5: Same-stem (from broad jsonl where same_stem_repo=True)
    same_stem_keys = set(make_key(r) for r in broad_data if r.get("same_stem_repo", False)) & pred_keys

    # Slice 6: Path-misled (from broad jsonl where path_misled_strict=True)
    path_misled_keys = set(make_key(r) for r in broad_data if r.get("path_misled_strict", False)) & pred_keys

    # Report
    slices = [
        ("Full test", full_keys),
        ("Code-Crucial v2 strict", strict_keys),
        ("Code-Crucial v2 broad", broad_keys),
        ("Low Jaccard (Q1)", low_jacc_keys),
        ("Same-stem", same_stem_keys),
        ("Path-misled", path_misled_keys),
    ]

    print(f"Predictions file: {args.predictions}")
    print(f"Total predictions loaded: {len(predictions)}")
    print(f"Total test examples: {len(test_data)}")
    print()

    header = f"{'Slice':<30s} {'N':>6s} {'R@1':>8s}"
    print(header)
    print("-" * len(header))

    for label, keys in slices:
        r1, count = compute_slice_r1(predictions, keys)
        print(f"{label:<30s} {count:>6d} {r1:>8.4f}")

    print()

    # Additional: Jaccard quartile stats
    if jaccards:
        print(f"Jaccard Q25 threshold: {q25:.4f}")
        print(f"Jaccard range: [{jacc_values.min():.4f}, {jacc_values.max():.4f}]")


if __name__ == "__main__":
    main()
