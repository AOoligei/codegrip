"""
Analyze why code-centric helps on mismatch slice but not aggregate.

Splits examples by issue-path Jaccard quartile and compares two models.

Usage:
    python scripts/analyze_mismatch_offset.py \
        --pred_pathonly <dir>/predictions.jsonl \
        --pred_codeaware <dir>/predictions.jsonl \
        --test_data data/grepo_text/grepo_test.jsonl

Deterministic (seed 42), CPU-only, standard libraries + numpy.
"""

import argparse
import json
import re
import numpy as np


def load_predictions(path):
    """Load predictions.jsonl, return dict keyed by (repo, issue_id_str)."""
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            data[key] = rec
    return data


def load_test_data(path):
    """Load test data, return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize_path(path):
    """Tokenize a file path into a set of tokens."""
    # Split on / . _ - and lowercase
    parts = re.split(r'[/._\-]', path.lower())
    return set(p for p in parts if p)


def compute_jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_issue_path_jaccard(issue_text, changed_files):
    """Compute Jaccard between issue text tokens and changed file path tokens."""
    # Issue text tokens: simple word tokenization
    issue_tokens = set(re.findall(r'[a-zA-Z0-9]+', issue_text.lower()))
    # Path tokens: union of all changed file path tokens
    path_tokens = set()
    for f in changed_files:
        path_tokens |= tokenize_path(f)
    return compute_jaccard(issue_tokens, path_tokens)


def compute_hit_at_1(rec):
    """Compute hit@1."""
    gt = set(rec["ground_truth"])
    predicted = rec["predicted"]
    if len(predicted) == 0:
        return 0.0
    return 1.0 if predicted[0] in gt else 0.0


def main():
    parser = argparse.ArgumentParser(description="Mismatch offset analysis")
    parser.add_argument("--pred_pathonly", required=True)
    parser.add_argument("--pred_codeaware", required=True)
    parser.add_argument("--test_data", required=True)
    args = parser.parse_args()

    np.random.seed(42)

    # Load data
    test_data = load_test_data(args.test_data)
    pred_path = load_predictions(args.pred_pathonly)
    pred_code = load_predictions(args.pred_codeaware)

    # Compute Jaccard for each test example
    examples = []
    for rec in test_data:
        key = (rec["repo"], str(rec["issue_id"]))
        if key not in pred_path or key not in pred_code:
            continue
        jaccard = compute_issue_path_jaccard(rec["issue_text"], rec["changed_files"])
        hit_path = compute_hit_at_1(pred_path[key])
        hit_code = compute_hit_at_1(pred_code[key])
        examples.append({
            "key": key,
            "jaccard": jaccard,
            "hit_path": hit_path,
            "hit_code": hit_code,
        })

    n = len(examples)
    print(f"Total aligned examples: {n}")
    print()

    # Sort by Jaccard and split into quartiles
    examples.sort(key=lambda x: x["jaccard"])
    jaccards = np.array([e["jaccard"] for e in examples])
    q25, q50, q75 = np.percentile(jaccards, [25, 50, 75])

    quartile_bounds = [
        ("Q1 (lowest)", 0.0, q25),
        ("Q2", q25, q50),
        ("Q3", q50, q75),
        ("Q4 (highest)", q75, 1.01),
    ]

    # Print quartile analysis
    header = f"{'Quartile':<16s} {'Jaccard range':<20s} {'N':>5s} {'R@1 path':>10s} {'R@1 code':>10s} {'Diff':>8s}"
    print(header)
    print("-" * len(header))

    for label, lo, hi in quartile_bounds:
        if label == "Q1 (lowest)":
            subset = [e for e in examples if e["jaccard"] <= q25]
        elif label == "Q4 (highest)":
            subset = [e for e in examples if e["jaccard"] > q75]
        elif label == "Q2":
            subset = [e for e in examples if q25 < e["jaccard"] <= q50]
        else:
            subset = [e for e in examples if q50 < e["jaccard"] <= q75]

        if not subset:
            continue
        cnt = len(subset)
        r1_path = np.mean([e["hit_path"] for e in subset])
        r1_code = np.mean([e["hit_code"] for e in subset])
        diff = r1_code - r1_path
        print(f"{label:<16s} [{lo:.3f}, {hi:.3f}){' ':>3s} {cnt:>5d} {r1_path:>10.4f} {r1_code:>10.4f} {diff:>+8.4f}")

    print()

    # Win/loss/tie analysis
    wins = sum(1 for e in examples if e["hit_code"] > e["hit_path"])
    losses = sum(1 for e in examples if e["hit_code"] < e["hit_path"])
    ties = sum(1 for e in examples if e["hit_code"] == e["hit_path"])

    print("Per-example win/loss/tie (code-aware vs path-only):")
    print(f"  Code-aware wins:  {wins:>5d} ({100*wins/n:.1f}%)")
    print(f"  Code-aware loses: {losses:>5d} ({100*losses/n:.1f}%)")
    print(f"  Ties:             {ties:>5d} ({100*ties/n:.1f}%)")
    print()

    # Breakdown by quartile
    print("Win/loss by quartile:")
    print(f"{'Quartile':<16s} {'Wins':>6s} {'Losses':>8s} {'Ties':>6s} {'Net':>6s}")
    print("-" * 46)

    for label, lo, hi in quartile_bounds:
        if label == "Q1 (lowest)":
            subset = [e for e in examples if e["jaccard"] <= q25]
        elif label == "Q4 (highest)":
            subset = [e for e in examples if e["jaccard"] > q75]
        elif label == "Q2":
            subset = [e for e in examples if q25 < e["jaccard"] <= q50]
        else:
            subset = [e for e in examples if q50 < e["jaccard"] <= q75]

        if not subset:
            continue
        w = sum(1 for e in subset if e["hit_code"] > e["hit_path"])
        l = sum(1 for e in subset if e["hit_code"] < e["hit_path"])
        t = sum(1 for e in subset if e["hit_code"] == e["hit_path"])
        print(f"{label:<16s} {w:>6d} {l:>8d} {t:>6d} {w-l:>+6d}")

    # Overall stats
    print()
    print(f"Overall R@1 path-only:  {np.mean([e['hit_path'] for e in examples]):.4f}")
    print(f"Overall R@1 code-aware: {np.mean([e['hit_code'] for e in examples]):.4f}")
    print(f"Jaccard quartile boundaries: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")


if __name__ == "__main__":
    main()
