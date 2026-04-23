#!/usr/bin/env python3
"""
Fast alpha sweep for decomposed reranker fusion.
Reads cached per-example scores from path-only and code-residual models,
computes fused score = s_path + alpha * s_code, evaluates R@1 on val set.

Usage:
    python scripts/alpha_sweep_fusion.py \
        --path_predictions /data/.../val_eval/path_only/predictions.jsonl \
        --code_predictions /data/.../val_eval/code_residual_v1/predictions.jsonl \
        --alpha 1.0

For autoresearch integration, prints a single line: "hit@1: XX.XX"
"""
import argparse
import json
import numpy as np


def load_predictions(path):
    """Load predictions.jsonl -> dict of (repo, issue_id) -> {candidate: score}"""
    preds = {}
    for line in open(path):
        d = json.loads(line)
        key = (d['repo'], d['issue_id'])
        candidates = d.get('predicted', d.get('candidates', []))
        scores_list = d.get('scores', [])
        scores = {c: s for c, s in zip(candidates, scores_list)}
        preds[key] = {
            'scores': scores,
            'gt': set(d.get('ground_truth', d.get('gt_files', d.get('changed_py_files', [])))),
            'candidates': candidates,
        }
    return preds


def fuse_and_eval(path_preds, code_preds, alpha):
    """Fuse scores and compute R@1."""
    hits = []
    for key in path_preds:
        if key not in code_preds:
            continue

        p = path_preds[key]
        c = code_preds[key]
        gt = p['gt']
        if not gt:
            continue

        # Fuse scores: s_total = s_path + alpha * s_code
        fused = {}
        for cand in p['candidates']:
            s_path = p['scores'].get(cand, 0.0)
            s_code = c['scores'].get(cand, 0.0)
            fused[cand] = s_path + alpha * s_code

        # Rank by fused score
        ranked = sorted(fused.keys(), key=lambda x: -fused[x])

        # R@1 = fraction of GT files in top-1 / total GT files (same as main eval)
        top_1 = set(ranked[:1])
        hit = len(top_1 & gt) / len(gt)
        hits.append(hit)

    return np.mean(hits) * 100 if hits else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_predictions', required=True)
    parser.add_argument('--code_predictions', required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep alpha from 0 to 3 in 0.1 steps')
    args = parser.parse_args()

    path_preds = load_predictions(args.path_predictions)
    code_preds = load_predictions(args.code_predictions)

    common = set(path_preds.keys()) & set(code_preds.keys())
    print(f"# Loaded {len(path_preds)} path, {len(code_preds)} code, {len(common)} common examples")

    if args.sweep:
        print(f"\n{'alpha':>8} {'R@1':>8}")
        print("-" * 20)
        best_alpha, best_r1 = 0.0, 0.0
        for alpha in np.arange(0.0, 3.05, 0.1):
            r1 = fuse_and_eval(path_preds, code_preds, alpha)
            marker = " *" if r1 > best_r1 else ""
            print(f"{alpha:>8.1f} {r1:>8.2f}{marker}")
            if r1 > best_r1:
                best_r1 = r1
                best_alpha = alpha
        print(f"\nBest: alpha={best_alpha:.1f}, R@1={best_r1:.2f}%")
        print(f"Path-only (alpha=0): R@1={fuse_and_eval(path_preds, code_preds, 0.0):.2f}%")
        print(f"Code-only (alpha=inf approx): R@1={fuse_and_eval(path_preds, code_preds, 1000.0):.2f}%")
    else:
        r1 = fuse_and_eval(path_preds, code_preds, args.alpha)
        # autoresearch-compatible output
        print(f"hit@1: {r1:.2f}")


if __name__ == '__main__':
    main()
