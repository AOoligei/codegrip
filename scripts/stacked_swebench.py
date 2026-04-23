#!/usr/bin/env python3
"""
Stacked ensemble for SWE-bench: train a meta-classifier over multiple
reranker outputs to select the best top-1.

Idea: 5 reranker checkpoints each produce ranked predictions. For each
(issue, candidate) pair, concatenate their ranks + scores as features.
Train a classifier to predict 'is this GT?'.

Labels come from SWE-bench train set (1874 examples with GT).

Usage:
    python scripts/stacked_swebench.py \
        --train_preds path1 path2 ... \
        --test_preds path1 path2 ... \
        --output_dir /data/chenlibin/grepo_agent_experiments/swebench_stacked
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec.get("repo", ""), str(rec.get("issue_id", "")))
            preds[key] = rec
    return preds


def build_features(preds_list, key, candidate):
    """For a given candidate, extract features from each model's ranking."""
    features = []
    for preds in preds_list:
        if key not in preds:
            features.extend([0.0, 0.0, 0.0])
            continue
        rec = preds[key]
        ranked = rec.get("predicted", [])
        scores = rec.get("scores", [])
        if candidate in ranked:
            rank = ranked.index(candidate)
            features.append(float(rank))
            features.append(1.0 / (1.0 + rank))  # reciprocal rank
            if scores and rank < len(scores):
                features.append(float(scores[rank]))
            else:
                features.append(0.0)
        else:
            features.extend([999.0, 0.0, -999.0])
    return features


def build_dataset(preds_list):
    """Build (X, y, keys, candidates) for stacked classifier."""
    X, y, keys, cands = [], [], [], []

    # Find common keys
    common = set(preds_list[0].keys())
    for p in preds_list[1:]:
        common &= set(p.keys())

    for key in common:
        rec = preds_list[0][key]
        gt_files = set(rec.get("ground_truth", []))
        if not gt_files:
            continue

        # Union of all predicted candidates across models (top-20 each)
        cand_set = set()
        for p in preds_list:
            if key in p:
                cand_set.update(p[key].get("predicted", [])[:20])

        for c in cand_set:
            feat = build_features(preds_list, key, c)
            label = 1 if c in gt_files else 0
            X.append(feat)
            y.append(label)
            keys.append(key)
            cands.append(c)

    return np.array(X), np.array(y), keys, cands


def evaluate_stacked(clf, preds_list, val_split=False):
    """Predict top-1 per example using classifier, compute h@1."""
    common = set(preds_list[0].keys())
    for p in preds_list[1:]:
        common &= set(p.keys())

    hits = []
    for key in sorted(common):
        rec = preds_list[0][key]
        gt_files = set(rec.get("ground_truth", []))
        if not gt_files:
            continue

        cand_set = set()
        for p in preds_list:
            if key in p:
                cand_set.update(p[key].get("predicted", [])[:20])

        cand_list = list(cand_set)
        if not cand_list:
            continue

        X = np.array([build_features(preds_list, key, c) for c in cand_list])
        probs = clf.predict_proba(X)[:, 1]
        best_idx = int(np.argmax(probs))
        top1 = cand_list[best_idx]

        hits.append(1.0 if top1 in gt_files else 0.0)

    return np.mean(hits) * 100 if hits else 0.0, len(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_preds", type=str, nargs="+", required=True)
    parser.add_argument("--test_preds", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading train predictions...")
    train_preds_list = [load_predictions(p) for p in args.train_preds]
    for p, pd in zip(args.train_preds, train_preds_list):
        print(f"  {os.path.basename(os.path.dirname(p))}: {len(pd)} examples")

    print("\nLoading test predictions...")
    test_preds_list = [load_predictions(p) for p in args.test_preds]
    for p, pd in zip(args.test_preds, test_preds_list):
        print(f"  {os.path.basename(os.path.dirname(p))}: {len(pd)} examples")

    print("\nBuilding training features...")
    X_train, y_train, _, _ = build_dataset(train_preds_list)
    print(f"  Train: {len(X_train)} (candidate,issue) pairs, "
          f"{y_train.sum()} positives ({y_train.mean()*100:.1f}%)")

    print("\nTraining classifier (GBM)...")
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    print(f"  Train AUC: {train_auc:.4f}")

    # Evaluate on train
    train_r1, n_train_issues = evaluate_stacked(clf, train_preds_list)
    print(f"\n=== Train performance (sanity check) ===")
    print(f"  Stacked h@1 = {train_r1:.2f}% (n={n_train_issues})")

    # Evaluate on test
    test_r1, n_test_issues = evaluate_stacked(clf, test_preds_list)
    print(f"\n=== Test performance ===")
    print(f"  Stacked h@1 = {test_r1:.2f}% (n={n_test_issues})")

    # Individual test baselines
    print(f"\n=== Individual test h@1 ===")
    common_test = set(test_preds_list[0].keys())
    for p in test_preds_list[1:]:
        common_test &= set(p.keys())

    individual_r1 = {}
    for i, (path, preds) in enumerate(zip(args.test_preds, test_preds_list)):
        hits = []
        for key in common_test:
            rec = preds[key]
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            if gt:
                hits.append(1.0 if (set(predicted[:1]) & gt) else 0.0)
        r1 = np.mean(hits) * 100 if hits else 0.0
        label = os.path.basename(os.path.dirname(path))
        individual_r1[label] = r1
        print(f"  {label}: {r1:.2f}%")

    summary = {
        "train_auc": float(train_auc),
        "train_stacked_h@1": float(train_r1),
        "test_stacked_h@1": float(test_r1),
        "n_train_issues": n_train_issues,
        "n_test_issues": n_test_issues,
        "individual_test_h@1": individual_r1,
        "swerank_reference": 56.3,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
