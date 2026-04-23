#!/usr/bin/env python3
"""
Train SPECTER router on TRAINING set (held-out evaluation).

Unlike the OOF CV router in eval_specter.py, this trains on training set
path predictions and deploys on test set. Gives a proper held-out number.

Input: path expert's training-set predictions (needs to be precomputed)
Output: trained router (joblib) for use in eval_specter.py

Usage:
    python scripts/train_specter_router.py \
        --train_preds path/to/train_predictions.jsonl \
        --output_dir /data/chenlibin/grepo_agent_experiments/specter/router
"""

import argparse
import json
import os
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib


def path_coverage(issue_text, path):
    issue_tokens = set(t for t in re.split(r'[/._\-\s,;:!?()\[\]{}"\'`<>]+',
                                            issue_text.lower()) if t)
    path_tokens = set(t for t in re.split(r'[/._\-]', path.lower()) if t)
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def has_dup_stem(candidates, top_k=10):
    stems = []
    for c in candidates[:top_k]:
        stem = os.path.splitext(os.path.basename(c))[0]
        stem = re.sub(r'^test_|_test$|^tests_|_tests$', '', stem)
        stems.append(stem)
    return len(stems) != len(set(stems))


def extract_features(issue_text, candidates, scores):
    if len(scores) < 2:
        score_gap = 0
        score_std = 0
        top1_score = 0
    else:
        score_gap = scores[0] - scores[1]
        score_std = float(np.std(scores[:10]))
        top1_score = scores[0]
    top1_path = candidates[0] if candidates else ""
    max_cov = max(path_coverage(issue_text, p) for p in candidates[:5]) if candidates else 0
    top1_cov = path_coverage(issue_text, top1_path)
    dup = 1 if has_dup_stem(candidates) else 0
    issue_len = len(issue_text.split())
    return [score_gap, score_std, top1_score, max_cov, top1_cov, dup, issue_len, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_preds", type=str, required=True,
                        help="JSONL with training-set path predictions")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading training predictions from {args.train_preds}")
    features = []
    labels = []
    with open(args.train_preds) as f:
        for line in f:
            rec = json.loads(line)
            issue = rec["issue_text"]
            ranked_cands = rec["ranked_candidates"]
            scores = rec["scores"]
            gt = set(rec["ground_truth"])

            feat = extract_features(issue, ranked_cands, scores)
            # Label: 1 if path top-1 is wrong
            label = 0 if ranked_cands[0] in gt else 1
            features.append(feat)
            labels.append(label)

    X = np.array(features)
    y = np.array(labels)
    print(f"  {len(X)} examples, {y.sum()} path-wrong ({y.mean()*100:.1f}%)")

    print("\nTraining router...")
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    clf.fit(X, y)

    # Training AUC
    train_prob = clf.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, train_prob)
    print(f"  Training AUC: {train_auc:.4f}")

    # Save
    out_path = os.path.join(args.output_dir, "router.joblib")
    joblib.dump(clf, out_path)
    print(f"\nSaved router to {out_path}")

    summary = {
        "num_train_examples": len(X),
        "train_path_wrong_rate": float(y.mean()),
        "train_auc": float(train_auc),
        "feature_names": [
            "score_gap", "score_std", "top1_score",
            "max_cov", "top1_cov", "dup_stem", "issue_len", "reserved",
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
