"""Analyze phantom hits: how many top-ranked files are completely irrelevant
(different directory from any GT file) across scales and pools."""

import json
import os
import numpy as np

def load_predictions(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_dir(filepath):
    """Get directory component of a file path."""
    return os.path.dirname(filepath)

def analyze_phantom_hits(preds, top_k=5):
    """Count how many top-K predictions share NO directory with any GT file."""
    phantom_counts = []
    path_overlap_counts = []

    for p in preds:
        gt = set(p['ground_truth']) if isinstance(p['ground_truth'], list) else {p['ground_truth']}
        gt_dirs = set(get_dir(f) for f in gt)
        predicted = p['predicted'][:top_k]

        phantoms = 0
        path_overlaps = 0
        for f in predicted:
            f_dir = get_dir(f)
            if f_dir not in gt_dirs and f not in gt:
                phantoms += 1
            # Check path token overlap with GT
            f_tokens = set(f.replace('/', '_').replace('.', '_').split('_'))
            gt_tokens = set()
            for gf in gt:
                gt_tokens.update(gf.replace('/', '_').replace('.', '_').split('_'))
            if f_tokens & gt_tokens and f not in gt:
                path_overlaps += 1

        phantom_counts.append(phantoms / top_k)
        path_overlap_counts.append(path_overlaps / top_k)

    return {
        "phantom_rate": np.mean(phantom_counts),
        "path_overlap_distractor_rate": np.mean(path_overlap_counts),
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

    print("Phantom Hit Analysis (Top-5)")
    print("=" * 70)
    print(f"{'Config':<16} {'Phantom Rate':>15} {'Path-Overlap Distractor':>25}")
    print("-" * 70)

    for name in ["0.5B-graph", "0.5B-hybrid", "1.5B-graph", "1.5B-hybrid",
                  "3B-graph", "3B-hybrid", "7B-graph", "7B-hybrid"]:
        path = configs[name]
        if not os.path.exists(path):
            continue
        preds = load_predictions(path)
        stats = analyze_phantom_hits(preds, top_k=5)
        print(f"{name:<16} {stats['phantom_rate']*100:>14.1f}% {stats['path_overlap_distractor_rate']*100:>24.1f}%")

if __name__ == "__main__":
    main()
