"""Phase diagram: R@1 vs top-K budget for different model scales and pools.
Recomputes metrics from existing predictions at varying budget cuts."""

import json
import os
import sys
from collections import defaultdict

def load_predictions(path):
    """Load predictions.jsonl and return list of dicts."""
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line))
    return preds

def compute_recall_at_k(preds, budget, k=1):
    """Given predictions, restrict to top-`budget` candidates, then compute R@k."""
    total = 0
    hits = 0
    for p in preds:
        gt = set(p['ground_truth']) if isinstance(p['ground_truth'], list) else {p['ground_truth']}
        predicted = p['predicted'][:budget]  # restrict to budget
        top_k = predicted[:k]
        # partial recall: fraction of GT in top-k
        if len(gt) > 0:
            recall = len(set(top_k) & gt) / len(gt)
            hits += recall
            total += 1
    return (hits / total * 100) if total > 0 else 0.0

def main():
    base = "/home/chenlibin/grepo_agent/experiments"

    # Define all configurations to evaluate
    configs = {
        # Graph-trained models on graph pool
        "0.5B-graph": f"{base}/scale_0.5B_graph/eval_merged_rerank/predictions.jsonl",
        "1.5B-graph": f"{base}/scale_1.5B_graph/eval_merged_rerank/predictions.jsonl",
        "3B-graph": f"{base}/scale_3B_graph/eval_merged_rerank/predictions.jsonl",
        "7B-graph": f"{base}/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
        # Graph-trained models on hybrid pool
        "0.5B-hybrid": f"{base}/scale_0.5B_graph/eval_hybrid/predictions.jsonl",
        "1.5B-hybrid": f"{base}/scale_1.5B_graph/eval_hybrid/predictions.jsonl",
        "3B-hybrid": f"{base}/scale_3B_graph/eval_hybrid/predictions.jsonl",
        "7B-hybrid": f"{base}/rankft_runB_graph/eval_hybrid_graph/predictions.jsonl",
    }

    budgets = [10, 20, 50, 100, 150, 200]

    print("Phase Diagram: R@1 vs Top-K Budget")
    print("=" * 80)

    # Header
    header = f"{'Config':<20}" + "".join(f"{'K='+str(b):>10}" for b in budgets)
    print(header)
    print("-" * len(header))

    results = {}
    for name, path in sorted(configs.items()):
        if not os.path.exists(path):
            print(f"{name:<20} FILE NOT FOUND: {path}")
            continue
        preds = load_predictions(path)
        row = []
        for b in budgets:
            r1 = compute_recall_at_k(preds, budget=b, k=1)
            row.append(r1)
        results[name] = row
        print(f"{name:<20}" + "".join(f"{r:>10.2f}" for r in row))

    # Also compute R@5 for the full phase diagram
    print()
    print("Phase Diagram: R@5 vs Top-K Budget")
    print("=" * 80)
    header = f"{'Config':<20}" + "".join(f"{'K='+str(b):>10}" for b in budgets)
    print(header)
    print("-" * len(header))

    for name, path in sorted(configs.items()):
        if not os.path.exists(path):
            continue
        preds = load_predictions(path)
        row = []
        for b in budgets:
            r5 = compute_recall_at_k(preds, budget=b, k=5)
            row.append(r5)
        print(f"{name:<20}" + "".join(f"{r:>10.2f}" for r in row))

    # Compute delta (hybrid - graph) at each budget
    print()
    print("Delta (Hybrid - Graph) R@1 at each budget")
    print("=" * 80)
    header = f"{'Scale':<20}" + "".join(f"{'K='+str(b):>10}" for b in budgets)
    print(header)
    print("-" * len(header))

    for scale in ["0.5B", "1.5B", "3B", "7B"]:
        g_key = f"{scale}-graph"
        h_key = f"{scale}-hybrid"
        if g_key in results and h_key in results:
            deltas = [h - g for g, h in zip(results[g_key], results[h_key])]
            print(f"{scale:<20}" + "".join(f"{d:>+10.2f}" for d in deltas))

    # Save as JSON for plotting
    output = {
        "budgets": budgets,
        "results": {k: v for k, v in results.items()},
    }
    out_path = f"{base}/../analysis/phase_diagram.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
