#!/usr/bin/env python3
"""
Code-Crucial subset sensitivity analysis.

Tests whether the Code-Crucial finding (code-residual underperforms path-only)
is robust to different construction definitions:
1. Original: path overlap < 0.1 AND code-residual model retrieves GT
2. Variant A: path overlap < 0.05 (stricter)
3. Variant B: path overlap < 0.2 (looser)
4. Variant C: No witness requirement, just low overlap
5. Variant D: Different overlap metrics (Jaccard vs token fraction)

Reports path-only vs code-residual R@1 for each definition.

Usage:
    python scripts/code_crucial_sensitivity.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/code_crucial_sensitivity
"""

import argparse
import json
import os
import re

import numpy as np

np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"

# Prediction files
PATH_ONLY_PREDS = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl"
CODE_RESIDUAL_PREDS = "/data/chenlibin/grepo_agent_experiments/code_residual_7b_v2/eval_graph/predictions.jsonl"


def load_predictions(path):
    """Load predictions keyed by (repo, issue_id)."""
    preds = {}
    if not os.path.isfile(path):
        print(f"  WARNING: {path} not found")
        return preds
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            hit1 = len(set(predicted[:1]) & gt) / max(1, len(gt))
            preds[key] = {
                "hit@1": hit1,
                "predicted": predicted,
                "ground_truth": gt,
            }
    return preds


def compute_overlap_fraction(issue_text, gt_files):
    """Fraction of path tokens that appear in issue text."""
    issue_tokens = set(re.split(r'\W+', issue_text.lower()))
    path_tokens = set()
    for f in gt_files:
        path_tokens.update(re.split(r'[/._\-]', f.lower()))
    path_tokens.discard('')
    issue_tokens.discard('')
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def compute_jaccard(issue_text, gt_files):
    """Jaccard similarity between issue tokens and path tokens."""
    issue_tokens = set(re.split(r'\W+', issue_text.lower()))
    path_tokens = set()
    for f in gt_files:
        path_tokens.update(re.split(r'[/._\-]', f.lower()))
    path_tokens.discard('')
    issue_tokens.discard('')
    union = issue_tokens | path_tokens
    if not union:
        return 0.0
    return len(issue_tokens & path_tokens) / len(union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/code_crucial_sensitivity")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            test_data[key] = rec

    path_preds = load_predictions(PATH_ONLY_PREDS)
    code_preds = load_predictions(CODE_RESIDUAL_PREDS)
    print(f"  {len(test_data)} test, {len(path_preds)} path, {len(code_preds)} code preds")

    # Build per-example features
    examples = []
    for key, rec in test_data.items():
        gt_files = rec.get("changed_py_files", rec.get("changed_files", []))
        if not gt_files or key not in path_preds:
            continue
        overlap_frac = compute_overlap_fraction(rec["issue_text"], gt_files)
        jaccard = compute_jaccard(rec["issue_text"], gt_files)
        path_hit = path_preds[key]["hit@1"]
        code_hit = code_preds[key]["hit@1"] if key in code_preds else 0
        code_retrieves = False
        if key in code_preds:
            code_retrieves = bool(
                code_preds[key]["ground_truth"] & set(code_preds[key]["predicted"][:20]))
        examples.append({
            "overlap_frac": overlap_frac,
            "jaccard": jaccard,
            "path_hit": path_hit,
            "code_hit": code_hit,
            "code_retrieves": code_retrieves,
        })

    print(f"  {len(examples)} examples")

    # Sensitivity analysis
    configs = [
        ("Original (frac<0.1, witness)", "overlap_frac", 0.1, True),
        ("Strict (frac<0.05, witness)", "overlap_frac", 0.05, True),
        ("Loose (frac<0.2, witness)", "overlap_frac", 0.2, True),
        ("No witness (frac<0.1)", "overlap_frac", 0.1, False),
        ("No witness (frac<0.05)", "overlap_frac", 0.05, False),
        ("No witness (frac<0.2)", "overlap_frac", 0.2, False),
        ("Jaccard<0.05, witness", "jaccard", 0.05, True),
        ("Jaccard<0.1, witness", "jaccard", 0.1, True),
        ("Jaccard<0.05, no witness", "jaccard", 0.05, False),
    ]

    results = []
    print(f"\n{'Definition':<35} {'N':>5} {'Path':>8} {'Code':>8} {'Delta':>8}")
    print("-" * 68)

    for name, metric, thresh, need_witness in configs:
        subset = [e for e in examples
                  if e[metric] < thresh
                  and (not need_witness or e["code_retrieves"])]
        if len(subset) < 10:
            print(f"{name:<35} {len(subset):>5} {'---':>8} {'---':>8} {'---':>8}")
            continue
        p = np.mean([e["path_hit"] for e in subset]) * 100
        c = np.mean([e["code_hit"] for e in subset]) * 100
        d = c - p
        print(f"{name:<35} {len(subset):>5} {p:>7.1f}% {c:>7.1f}% {d:>+7.1f}")
        results.append({"name": name, "n": len(subset),
                        "path_R@1": p, "code_R@1": c, "delta": d})

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
