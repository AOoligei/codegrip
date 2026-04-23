#!/usr/bin/env python3
"""
Path predictability analysis: how much can a simple model predict bug location
from path tokens alone vs path structure alone vs issue-path interaction?

Uses grouped cross-validation (by issue) to avoid data leakage, with
TF-IDF features inside the CV pipeline.

This quantifies the paper's claim that file paths carry strong signal about
bug location, even without any code content or neural model.

Usage:
    python scripts/mutual_information_analysis.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/mi_analysis
"""

import argparse
import json
import os
import random
import re

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

random.seed(42)
np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"


def tokenize_path(path: str) -> str:
    """Tokenize a file path into space-separated tokens."""
    tokens = re.split(r'[/._\-]', path)
    return " ".join(t.lower() for t in tokens if t)


def load_data(max_examples: int = 500):
    """Load test data and create (issue, candidate, label) triples with group IDs."""
    test_data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            test_data[key] = rec

    triples = []
    group_ids = []
    group_counter = 0

    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], rec["issue_id"])
            if key not in test_data:
                continue
            test_rec = test_data[key]
            gt_files = set(test_rec.get("changed_py_files",
                                        test_rec.get("changed_files", [])))
            candidates = rec.get("bm25_candidates", rec.get("candidates", []))[:50]

            for cand in candidates:
                label = 1 if cand in gt_files else 0
                triples.append({
                    "issue_text": test_rec["issue_text"][:2000],
                    "file_path": cand,
                    "label": label,
                    "repo": test_rec["repo"],
                    "issue_id": test_rec["issue_id"],
                })
                group_ids.append(group_counter)

            group_counter += 1
            if group_counter >= max_examples:
                break

    return triples, np.array(group_ids)


def grouped_cv_auc(X, y, groups, n_splits=5):
    """Compute AUC with GroupKFold (no issue leakage between folds)."""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    return roc_auc_score(y, y_pred)


def compute_path_lexical_auc(triples, groups):
    """AUC using path-issue lexical overlap features."""
    labels = np.array([t["label"] for t in triples])

    features = []
    for t in triples:
        issue_tokens = set(t["issue_text"].lower().split())
        path_tokens = set(tokenize_path(t["file_path"]).split())
        overlap = len(issue_tokens & path_tokens)
        total = len(path_tokens) + 1
        features.append([overlap, overlap / total, len(path_tokens)])

    X = np.array(features)
    return grouped_cv_auc(X, labels, groups)


def compute_path_tfidf_auc(triples, groups):
    """AUC using TF-IDF on path tokens, fitted inside each fold."""
    labels = np.array([t["label"] for t in triples])
    path_texts = [tokenize_path(t["file_path"]) for t in triples]

    gkf = GroupKFold(n_splits=5)
    y_pred = np.zeros(len(labels))

    for train_idx, val_idx in gkf.split(path_texts, labels, groups):
        # Fit TF-IDF inside the fold
        tfidf = TfidfVectorizer(max_features=500)
        X_train = tfidf.fit_transform([path_texts[i] for i in train_idx])
        X_val = tfidf.transform([path_texts[i] for i in val_idx])

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, labels[train_idx])
        y_pred[val_idx] = clf.predict_proba(X_val)[:, 1]

    return roc_auc_score(labels, y_pred)


def compute_issue_path_tfidf_auc(triples, groups):
    """AUC using TF-IDF on concatenated issue+path tokens."""
    labels = np.array([t["label"] for t in triples])
    combined = [f"{t['issue_text']} {tokenize_path(t['file_path'])}" for t in triples]

    gkf = GroupKFold(n_splits=5)
    y_pred = np.zeros(len(labels))

    for train_idx, val_idx in gkf.split(combined, labels, groups):
        tfidf = TfidfVectorizer(max_features=1000)
        X_train = tfidf.fit_transform([combined[i] for i in train_idx])
        X_val = tfidf.transform([combined[i] for i in val_idx])

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, labels[train_idx])
        y_pred[val_idx] = clf.predict_proba(X_val)[:, 1]

    return roc_auc_score(labels, y_pred)


def compute_path_structure_auc(triples, groups):
    """AUC using only structural path features (no lexical content)."""
    labels = np.array([t["label"] for t in triples])

    features = []
    for t in triples:
        path = t["file_path"]
        depth = path.count("/")
        is_test = 1 if "test" in path.lower() else 0
        is_init = 1 if "__init__" in path else 0
        n_components = len(path.split("/"))
        filename_len = len(path.split("/")[-1])
        features.append([depth, is_test, is_init, n_components, filename_len])

    X = np.array(features)
    return grouped_cv_auc(X, labels, groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/mi_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    triples, groups = load_data(args.max_examples)
    n_issues = len(set(groups.tolist()))
    n_pos = sum(1 for t in triples if t["label"] == 1)
    n_neg = sum(1 for t in triples if t["label"] == 0)
    print(f"  {len(triples)} triples from {n_issues} issues ({n_pos} pos, {n_neg} neg)")
    print(f"  Base rate: {n_pos / len(triples) * 100:.1f}%")

    print("\n1. Path lexical overlap AUC...")
    auc_overlap = compute_path_lexical_auc(triples, groups)
    print(f"   AUC: {auc_overlap:.4f}")

    print("\n2. Path TF-IDF AUC...")
    auc_path_tfidf = compute_path_tfidf_auc(triples, groups)
    print(f"   AUC: {auc_path_tfidf:.4f}")

    print("\n3. Issue+Path TF-IDF AUC...")
    auc_combined = compute_issue_path_tfidf_auc(triples, groups)
    print(f"   AUC: {auc_combined:.4f}")

    print("\n4. Path structure only AUC...")
    auc_structure = compute_path_structure_auc(triples, groups)
    print(f"   AUC: {auc_structure:.4f}")

    summary = {
        "num_issues": n_issues,
        "num_triples": len(triples),
        "base_rate": n_pos / len(triples),
        "auc_path_lexical_overlap": auc_overlap,
        "auc_path_tfidf": auc_path_tfidf,
        "auc_issue_path_tfidf": auc_combined,
        "auc_path_structure": auc_structure,
        "cv_method": "GroupKFold(n_splits=5) by (repo, issue_id)",
        "note": "AUC measures predictive power of path features for bug location. "
                "Higher AUC = paths carry more discriminative signal. "
                "All vectorizers fitted inside CV folds to prevent leakage.",
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (grouped 5-fold CV, no leakage) ===")
    print(f"Path lexical overlap:  AUC = {auc_overlap:.4f}")
    print(f"Path TF-IDF:           AUC = {auc_path_tfidf:.4f}")
    print(f"Issue+Path TF-IDF:     AUC = {auc_combined:.4f}")
    print(f"Path structure only:   AUC = {auc_structure:.4f}")
    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
