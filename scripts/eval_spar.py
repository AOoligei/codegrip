#!/usr/bin/env python3
"""
SPAR: Scope-aware Path-Aware Reranking evaluation.

Combines three debiasing strategies into one pipeline:
  Stage 1: Path-only scorer ranks all candidates
  Stage 2: Confidence estimation (score gap between top-1 and top-2)
  Stage 3: Low-confidence examples are re-ranked by a code-aware model
           using selectively retrieved function snippets

The code-aware model can be:
  a) The selective-retrieval trained model (existing)
  b) The contrastive PathSwap-trained model (new)
  c) Any code-aware LoRA

Reports: overall R@1, Code-Crucial R@1, shuffle R@1, function-level metrics.

Usage:
    python scripts/eval_spar.py \
        --path_preds experiments/.../predictions.jsonl \
        --code_preds /data/.../predictions.jsonl \
        --route_fraction 0.3 \
        --output_dir /data/chenlibin/grepo_agent_experiments/spar_eval
"""

import argparse
import json
import os
import re

import numpy as np

np.random.seed(42)

TEST_PATH = "/home/chenlibin/grepo_agent/data/grepo_text/grepo_test.jsonl"


def load_predictions(path):
    """Load predictions with scores for routing."""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            gt = set(rec.get("ground_truth", []))
            predicted = rec.get("predicted", [])
            scores = rec.get("scores", [])
            hit1 = len(set(predicted[:1]) & gt) / max(1, len(gt))
            hit3 = len(set(predicted[:3]) & gt) / max(1, len(gt))
            # Score gap = confidence proxy
            gap = (scores[0] - scores[1]) if len(scores) >= 2 else 999.0
            preds[key] = {
                "hit1": hit1, "hit3": hit3,
                "gap": gap, "predicted": predicted,
                "gt": gt, "scores": scores,
            }
    return preds


def load_test_data():
    data = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["repo"], str(rec["issue_id"]))
            gt = rec.get("changed_py_files", rec.get("changed_files", []))
            issue = rec["issue_text"]
            # Compute path overlap for Code-Crucial classification
            issue_tokens = set(re.split(r"\W+", issue.lower()))
            path_tokens = set()
            for g in gt:
                path_tokens.update(re.split(r"[/._\-]", g.lower()))
            path_tokens.discard("")
            overlap = len(issue_tokens & path_tokens) / max(1, len(path_tokens))
            data[key] = {"overlap": overlap, "gt_funcs": rec.get("changed_functions", [])}
    return data


def evaluate_spar(path_preds, code_preds, test_data, route_fractions):
    """Evaluate SPAR at different routing thresholds."""
    common = sorted(set(path_preds.keys()) & set(code_preds.keys()))

    # Sort by path confidence (ascending = least confident first)
    by_confidence = sorted(common, key=lambda k: path_preds[k]["gap"])

    # Baselines
    path_r1 = np.mean([path_preds[k]["hit1"] for k in common]) * 100
    code_r1 = np.mean([code_preds[k]["hit1"] for k in common]) * 100

    # Code-Crucial keys (overlap < 0.1)
    cc_keys = [k for k in common if test_data.get(k, {}).get("overlap", 1) < 0.1]

    results = []
    for frac in route_fractions:
        n_route = int(len(by_confidence) * frac)
        routed = set(by_confidence[:n_route])

        # SPAR: path for confident, code for uncertain
        hits = []
        cc_hits = []
        for k in common:
            if k in routed:
                hits.append(code_preds[k]["hit1"])
            else:
                hits.append(path_preds[k]["hit1"])

            if k in cc_keys:
                if k in routed:
                    cc_hits.append(code_preds[k]["hit1"])
                else:
                    cc_hits.append(path_preds[k]["hit1"])

        overall = np.mean(hits) * 100
        cc_overall = np.mean(cc_hits) * 100 if cc_hits else 0

        # Routed subset stats
        if routed:
            routed_path = np.mean([path_preds[k]["hit1"] for k in routed]) * 100
            routed_code = np.mean([code_preds[k]["hit1"] for k in routed]) * 100
        else:
            routed_path = routed_code = 0

        results.append({
            "route_fraction": frac,
            "n_routed": n_route,
            "overall_R@1": overall,
            "delta_vs_path": overall - path_r1,
            "code_crucial_R@1": cc_overall,
            "routed_path_R@1": routed_path,
            "routed_code_R@1": routed_code,
        })

    return results, path_r1, code_r1, len(common), len(cc_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_preds", type=str, required=True,
                        help="Path-only model predictions.jsonl")
    parser.add_argument("--code_preds", type=str, nargs="+", required=True,
                        help="Code-aware model predictions.jsonl (can list multiple)")
    parser.add_argument("--code_labels", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.code_labels or [os.path.basename(os.path.dirname(p)) for p in args.code_preds]

    print("Loading data...")
    test_data = load_test_data()
    path_preds = load_predictions(args.path_preds)
    print(f"  Path-only: {len(path_preds)} predictions")

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    all_results = {}
    for code_path, label in zip(args.code_preds, labels):
        print(f"\n=== SPAR with {label} ===")
        code_preds = load_predictions(code_path)
        print(f"  Code model: {len(code_preds)} predictions")

        results, path_r1, code_r1, n, n_cc = evaluate_spar(
            path_preds, code_preds, test_data, fractions)

        print(f"  Path-only R@1: {path_r1:.2f}%")
        print(f"  Code-only R@1: {code_r1:.2f}%")
        print(f"  Examples: {n} total, {n_cc} Code-Crucial")
        print()
        print(f"  {'Route%':>7} {'Overall':>9} {'Delta':>7} {'CC-R@1':>8} {'Rt-Path':>8} {'Rt-Code':>8}")
        print(f"  {'-'*52}")

        for r in results:
            print(f"  {r['route_fraction']*100:>5.0f}%  "
                  f"{r['overall_R@1']:>8.2f}%  "
                  f"{r['delta_vs_path']:>+6.2f}  "
                  f"{r['code_crucial_R@1']:>7.2f}%  "
                  f"{r['routed_path_R@1']:>7.1f}%  "
                  f"{r['routed_code_R@1']:>7.1f}%")

        all_results[label] = {
            "path_r1": path_r1, "code_r1": code_r1,
            "n": n, "n_cc": n_cc, "routing": results,
        }

    with open(os.path.join(args.output_dir, "spar_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/spar_results.json")


if __name__ == "__main__":
    main()
