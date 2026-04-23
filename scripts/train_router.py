#!/usr/bin/env python3
"""
Train a selective router: decides when to trust path-only vs invoke code specialist.

Phase 1: Extract routing features from path-only predictions (CPU)
Phase 2: Train router (XGBoost/LogisticRegression) (CPU)
Phase 3: Evaluate routed system (CPU, uses cached predictions)

Usage:
    python scripts/train_router.py \
        --output_dir /data/chenlibin/grepo_agent_experiments/router
"""

import argparse
import json
import os
import re
import random

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

random.seed(42)
np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"
BM25_PATH = "/home/chenlibin/grepo_agent/data/rankft/merged_bm25_exp6_candidates.jsonl"
PATH_PREDS = "/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl"
CODE_PREDS = "/data/chenlibin/grepo_agent_experiments/hierarchical_eval/trained_baseline_k10/per_example.jsonl"


def load_test_data():
    data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            data[key] = rec
    return data


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec.get("issue_id", "")))
            preds[key] = rec
    return preds


def compute_path_overlap(issue_text, file_path):
    """Jaccard overlap between issue tokens and path tokens."""
    issue_tokens = set(re.split(r'\W+', issue_text.lower()))
    path_tokens = set(re.split(r'[/._\-]', file_path.lower()))
    path_tokens.discard('')
    if not path_tokens:
        return 0.0
    return len(issue_tokens & path_tokens) / len(path_tokens)


def has_duplicate_stem(candidates, top_k=10):
    """Check if any files in top-k share the same stem."""
    stems = []
    for c in candidates[:top_k]:
        stem = os.path.splitext(os.path.basename(c))[0]
        # Normalize: strip test_ prefix and _test suffix
        stem = stem.replace("test_", "").replace("_test", "")
        stems.append(stem)
    return len(stems) != len(set(stems))


def extract_features(test_data, path_preds, bm25_data):
    """Extract routing features for each example."""
    features = []
    labels = []
    keys = []
    groups = []
    repo_to_id = {}

    for key, pred in path_preds.items():
        if key not in test_data:
            continue

        test_rec = test_data[key]
        gt = set(pred.get("ground_truth", []))
        predicted = pred.get("predicted", [])
        scores = pred.get("scores", [])

        if not gt or not predicted:
            continue

        # Label: 1 if path-only top-1 is WRONG
        top1_hit = len(set(predicted[:1]) & gt) > 0
        label = 0 if top1_hit else 1

        # Features
        issue_text = test_rec["issue_text"]
        top1_path = predicted[0] if predicted else ""

        # Score-based features
        if len(scores) >= 2:
            score_gap = scores[0] - scores[1]
            score_std = float(np.std(scores[:10])) if len(scores) >= 10 else 0
            top1_score = scores[0]
        else:
            score_gap = 0
            score_std = 0
            top1_score = 0

        # Path overlap
        max_overlap = max(compute_path_overlap(issue_text, p)
                          for p in predicted[:5]) if predicted else 0
        top1_overlap = compute_path_overlap(issue_text, top1_path)

        # Ambiguity features
        dup_stem = 1 if has_duplicate_stem(predicted) else 0

        # Issue length
        issue_len = len(issue_text.split())

        # Num GT files (proxy: how many predicted are in GT)
        gt_in_top10 = len(set(predicted[:10]) & gt)

        feat = [
            score_gap,
            score_std,
            top1_score,
            max_overlap,
            top1_overlap,
            dup_stem,
            issue_len,
            gt_in_top10,
        ]

        features.append(feat)
        labels.append(label)
        keys.append(key)

        repo = key[0]
        if repo not in repo_to_id:
            repo_to_id[repo] = len(repo_to_id)
        groups.append(repo_to_id[repo])

    return np.array(features), np.array(labels), keys, np.array(groups)


def evaluate_routing(path_preds, code_preds, router_decisions, test_data):
    """Evaluate the routed system."""
    path_hits = 0
    routed_hits = 0
    oracle_hits = 0
    n_routed = 0
    total = 0

    # Code-Crucial keys (low overlap examples)
    cc_path_hits = 0
    cc_routed_hits = 0
    cc_total = 0

    for key, pred in path_preds.items():
        if key not in test_data:
            continue
        gt = set(pred.get("ground_truth", []))
        predicted = pred.get("predicted", [])
        if not gt or not predicted:
            continue

        # Use binary hit (any GT in top-1) to match code_preds format
        path_hit = 1.0 if len(set(predicted[:1]) & gt) > 0 else 0.0
        path_hits += path_hit

        # Code prediction (from hierarchical eval)
        code_hit = 0
        if key in code_preds:
            code_rec = code_preds[key]
            code_hit = code_rec.get("hier_hit", 0)

        # Routed decision
        route_to_code = router_decisions.get(key, False)
        if route_to_code:
            routed_hits += code_hit
            n_routed += 1
        else:
            routed_hits += path_hit

        # Oracle: pick whichever is better
        oracle_hits += max(path_hit, code_hit)
        total += 1

        # Code-Crucial slice (low overlap)
        test_rec = test_data[key]
        overlap = max(compute_path_overlap(test_rec["issue_text"], p)
                      for p in predicted[:5]) if predicted else 0
        if overlap < 0.1:
            cc_path_hits += path_hit
            cc_routed_hits += (code_hit if route_to_code else path_hit)
            cc_total += 1

    return {
        "total": total,
        "n_routed": n_routed,
        "route_fraction": n_routed / max(1, total),
        "path_only_R@1": path_hits / max(1, total) * 100,
        "routed_R@1": routed_hits / max(1, total) * 100,
        "oracle_R@1": oracle_hits / max(1, total) * 100,
        "cc_total": cc_total,
        "cc_path_R@1": cc_path_hits / max(1, cc_total) * 100 if cc_total else 0,
        "cc_routed_R@1": cc_routed_hits / max(1, cc_total) * 100 if cc_total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenlibin/grepo_agent_experiments/router")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    test_data = load_test_data()
    path_preds = load_predictions(PATH_PREDS)
    print(f"  {len(test_data)} test, {len(path_preds)} path predictions")

    # Load code predictions
    code_preds = {}
    if os.path.isfile(CODE_PREDS):
        with open(CODE_PREDS) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["repo"], rec["issue_id"])
                code_preds[key] = rec
        print(f"  {len(code_preds)} code predictions")

    # Load BM25 data for features
    bm25_data = {}
    with open(BM25_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            bm25_data[key] = rec

    print("\nPhase 1: Extracting routing features...")
    X, y, keys, groups = extract_features(test_data, path_preds, bm25_data)
    print(f"  {len(X)} examples, {y.sum()} path-wrong ({y.mean()*100:.1f}%)")
    print(f"  Features: score_gap, score_std, top1_score, max_overlap, "
          f"top1_overlap, dup_stem, issue_len, gt_in_top10")

    print("\nPhase 2: Training router (GroupKFold CV)...")
    gkf = GroupKFold(n_splits=5)
    y_pred_proba = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        y_pred_proba[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    auc = roc_auc_score(y, y_pred_proba)
    print(f"  Router AUC: {auc:.4f}")

    # Sweep routing thresholds
    print("\nPhase 3: Evaluating routing strategies...")
    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        decisions = {}
        for i, key in enumerate(keys):
            decisions[key] = y_pred_proba[i] > threshold

        r = evaluate_routing(path_preds, code_preds, decisions, test_data)
        r["threshold"] = threshold
        results.append(r)
        print(f"  t={threshold:.1f}: route={r['route_fraction']:.0%} "
              f"path={r['path_only_R@1']:.2f}% routed={r['routed_R@1']:.2f}% "
              f"oracle={r['oracle_R@1']:.2f}% "
              f"CC: path={r['cc_path_R@1']:.1f}% routed={r['cc_routed_R@1']:.1f}%")

    # Also report no-routing and oracle
    no_route = evaluate_routing(path_preds, code_preds,
                                {k: False for k in keys}, test_data)
    all_route = evaluate_routing(path_preds, code_preds,
                                 {k: True for k in keys}, test_data)
    # True oracle: route iff code_hit > path_hit for that example
    oracle_decisions = {}
    for i, key in enumerate(keys):
        path_hit = 1.0 if y[i] == 0 else 0.0  # y=0 means path correct
        code_hit = code_preds.get(key, {}).get("hier_hit", 0)
        oracle_decisions[key] = code_hit > path_hit
    oracle_route = evaluate_routing(path_preds, code_preds, oracle_decisions, test_data)

    print(f"\n  No routing:     R@1={no_route['routed_R@1']:.2f}%")
    print(f"  All to code:    R@1={all_route['routed_R@1']:.2f}%")
    print(f"  Oracle routing: R@1={oracle_route['routed_R@1']:.2f}%")

    summary = {
        "router_auc": auc,
        "n_examples": len(X),
        "path_wrong_rate": float(y.mean()),
        "threshold_sweep": results,
        "no_routing": no_route,
        "all_code": all_route,
        "oracle_routing": oracle_route,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
