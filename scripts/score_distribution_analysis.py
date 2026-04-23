"""Analyze score distributions across scales and pools.
Shows WHY 7B fails on hybrid: score gap collapse, distractor score inflation."""

import json
import os
import numpy as np
from collections import defaultdict

def load_predictions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def analyze_score_distribution(preds):
    """Compute per-example score statistics."""
    gt_scores = []
    top_neg_scores = []
    score_gaps = []
    gt_ranks = []

    for p in preds:
        gt = set(p['ground_truth']) if isinstance(p['ground_truth'], list) else {p['ground_truth']}
        predicted = p['predicted']
        scores = p['scores']

        # Find GT file scores and ranks
        for i, (f, s) in enumerate(zip(predicted, scores)):
            if f in gt:
                gt_scores.append(s)
                gt_ranks.append(i + 1)
                # Score gap: GT score minus top non-GT score
                non_gt_scores = [sc for ff, sc in zip(predicted, scores) if ff not in gt]
                if non_gt_scores:
                    score_gaps.append(s - max(non_gt_scores))
                    top_neg_scores.append(max(non_gt_scores))
                break  # just first GT file found

    return {
        "gt_score_mean": np.mean(gt_scores) if gt_scores else 0,
        "gt_score_std": np.std(gt_scores) if gt_scores else 0,
        "top_neg_score_mean": np.mean(top_neg_scores) if top_neg_scores else 0,
        "score_gap_mean": np.mean(score_gaps) if score_gaps else 0,
        "score_gap_median": np.median(score_gaps) if score_gaps else 0,
        "score_gap_negative_frac": np.mean([g < 0 for g in score_gaps]) if score_gaps else 0,
        "gt_rank_mean": np.mean(gt_ranks) if gt_ranks else 0,
        "gt_rank_median": np.median(gt_ranks) if gt_ranks else 0,
        "gt_found_frac": len(gt_scores) / len(preds) if preds else 0,
    }

def main():
    base = "/home/chenlibin/grepo_agent/experiments"

    configs = {
        "0.5B-graph": f"{base}/scale_0.5B_graph/eval_merged_rerank/predictions.jsonl",
        "1.5B-graph": f"{base}/scale_1.5B_graph/eval_merged_rerank/predictions.jsonl",
        "3B-graph": f"{base}/scale_3B_graph/eval_merged_rerank/predictions.jsonl",
        "7B-graph": f"{base}/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
        "0.5B-hybrid": f"{base}/scale_0.5B_graph/eval_hybrid/predictions.jsonl",
        "1.5B-hybrid": f"{base}/scale_1.5B_graph/eval_hybrid/predictions.jsonl",
        "3B-hybrid": f"{base}/scale_3B_graph/eval_hybrid/predictions.jsonl",
        "7B-hybrid": f"{base}/rankft_runB_graph/eval_hybrid_graph/predictions.jsonl",
    }

    print("Score Distribution Analysis: Why 7B Fails on Hybrid Pool")
    print("=" * 100)

    header = (f"{'Config':<16} {'GT Score':>10} {'Top Neg':>10} {'Gap Mean':>10} "
              f"{'Gap Med':>10} {'Gap<0 %':>10} {'GT Rank':>10} {'GT Found':>10}")
    print(header)
    print("-" * len(header))

    all_stats = {}
    for name in ["0.5B-graph", "0.5B-hybrid", "1.5B-graph", "1.5B-hybrid",
                  "3B-graph", "3B-hybrid", "7B-graph", "7B-hybrid"]:
        path = configs[name]
        if not os.path.exists(path):
            print(f"{name:<16} FILE NOT FOUND")
            continue
        preds = load_predictions(path)
        stats = analyze_score_distribution(preds)
        all_stats[name] = stats
        print(f"{name:<16} {stats['gt_score_mean']:>10.2f} {stats['top_neg_score_mean']:>10.2f} "
              f"{stats['score_gap_mean']:>10.2f} {stats['score_gap_median']:>10.2f} "
              f"{stats['score_gap_negative_frac']*100:>9.1f}% {stats['gt_rank_mean']:>10.1f} "
              f"{stats['gt_found_frac']*100:>9.1f}%")

        if name.endswith("-hybrid") and name.replace("-hybrid", "-graph") in all_stats:
            gname = name.replace("-hybrid", "-graph")
            gs = all_stats[gname]
            hs = stats
            scale = name.split("-")[0]
            print(f"  >> {scale} delta: gap_mean {hs['score_gap_mean']-gs['score_gap_mean']:+.2f}, "
                  f"gap<0% {(hs['score_gap_negative_frac']-gs['score_gap_negative_frac'])*100:+.1f}pp, "
                  f"rank {hs['gt_rank_mean']-gs['gt_rank_mean']:+.1f}")

    # Save
    out_path = "/home/chenlibin/grepo_agent/analysis/score_distributions.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
