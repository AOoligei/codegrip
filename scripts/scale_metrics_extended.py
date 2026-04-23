"""Extended scale analysis: R@1, R@3, R@5, R@10 across all scales and pools."""
import json, os, numpy as np

def load_preds(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def recall_at_k(preds, k):
    total = hits = 0
    for p in preds:
        gt = set(p['ground_truth']) if isinstance(p['ground_truth'], list) else {p['ground_truth']}
        top_k = p['predicted'][:k]
        if gt:
            hits += len(set(top_k) & gt) / len(gt)
            total += 1
    return hits / total * 100 if total else 0

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

ks = [1, 3, 5, 10, 20]
print(f"{'Config':<16}" + "".join(f"{'R@'+str(k):>8}" for k in ks))
print("-" * 60)

for name in ["0.5B-graph", "0.5B-hybrid", "1.5B-graph", "1.5B-hybrid",
              "3B-graph", "3B-hybrid", "7B-graph", "7B-hybrid"]:
    if not os.path.exists(configs[name]):
        print(f"{name:<16} NOT FOUND"); continue
    preds = load_preds(configs[name])
    vals = [recall_at_k(preds, k) for k in ks]
    print(f"{name:<16}" + "".join(f"{v:>8.2f}" for v in vals))

# Delta table
print("\nDelta (Hybrid - Graph)")
print(f"{'Scale':<16}" + "".join(f"{'R@'+str(k):>8}" for k in ks))
print("-" * 60)
for scale in ["0.5B", "1.5B", "3B", "7B"]:
    gp = configs[f"{scale}-graph"]
    hp = configs[f"{scale}-hybrid"]
    if os.path.exists(gp) and os.path.exists(hp):
        gp_preds = load_preds(gp)
        hp_preds = load_preds(hp)
        deltas = [recall_at_k(hp_preds, k) - recall_at_k(gp_preds, k) for k in ks]
        print(f"{scale:<16}" + "".join(f"{d:>+8.2f}" for d in deltas))
