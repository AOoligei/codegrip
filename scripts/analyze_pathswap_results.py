"""Analyze PathSwap-GREPO results: compare original vs pathswapped R@K across scales."""
import json, os, sys

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

# Original results
original = {
    "0.5B-graph": f"{base}/scale_0.5B_graph/eval_merged_rerank/predictions.jsonl",
    "1.5B-graph": f"{base}/scale_1.5B_graph/eval_merged_rerank/predictions.jsonl",
    "3B-graph": f"{base}/scale_3B_graph/eval_merged_rerank/predictions.jsonl",
    "7B-graph": f"{base}/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
    "0.5B-hybrid": f"{base}/scale_0.5B_graph/eval_hybrid/predictions.jsonl",
    "1.5B-hybrid": f"{base}/scale_1.5B_graph/eval_hybrid/predictions.jsonl",
    "3B-hybrid": f"{base}/scale_3B_graph/eval_hybrid/predictions.jsonl",
    "7B-hybrid": f"{base}/rankft_runB_graph/eval_hybrid_graph/predictions.jsonl",
}

# PathSwap results
pathswap = {
    "0.5B-graph": f"{base}/pathswap_eval/0.5B_graph/predictions.jsonl",
    "1.5B-graph": f"{base}/pathswap_eval/1.5B_graph/predictions.jsonl",
    "3B-graph": f"{base}/pathswap_eval/3B_graph/predictions.jsonl",
    "7B-graph": f"{base}/pathswap_eval/7B_graph/predictions.jsonl",
    "0.5B-hybrid": f"{base}/pathswap_eval/0.5B_hybrid/predictions.jsonl",
    "1.5B-hybrid": f"{base}/pathswap_eval/1.5B_hybrid/predictions.jsonl",
    "3B-hybrid": f"{base}/pathswap_eval/3B_hybrid/predictions.jsonl",
    "7B-hybrid": f"{base}/pathswap_eval/7B_hybrid/predictions.jsonl",
}

ks = [1, 3, 5, 10]

print("PathSwap-GREPO Counterfactual Analysis")
print("=" * 90)
print(f"{'Config':<16} {'Pool':<8}", end="")
for k in ks:
    print(f"  {'Orig R@'+str(k):>10} {'PS R@'+str(k):>10} {'Delta':>8}", end="")
print()
print("-" * 90)

results = {}
for scale in ["0.5B", "1.5B", "3B", "7B"]:
    for pool in ["graph", "hybrid"]:
        name = f"{scale}-{pool}"
        orig_path = original.get(name)
        ps_path = pathswap.get(name)

        if not orig_path or not os.path.exists(orig_path):
            continue
        if not ps_path or not os.path.exists(ps_path):
            continue

        orig_preds = load_preds(orig_path)
        ps_preds = load_preds(ps_path)

        print(f"{scale:<16} {pool:<8}", end="")
        row = {}
        for k in ks:
            orig_r = recall_at_k(orig_preds, k)
            ps_r = recall_at_k(ps_preds, k)
            delta = ps_r - orig_r
            print(f"  {orig_r:>10.2f} {ps_r:>10.2f} {delta:>+8.2f}", end="")
            row[f"orig_R@{k}"] = orig_r
            row[f"ps_R@{k}"] = ps_r
            row[f"delta_R@{k}"] = delta
        print()
        results[name] = row

# Summary: relative collapse
print("\nRelative Collapse (PathSwap R@1 / Original R@1)")
print("-" * 50)
for scale in ["0.5B", "1.5B", "3B", "7B"]:
    for pool in ["graph", "hybrid"]:
        name = f"{scale}-{pool}"
        if name in results:
            orig = results[name]["orig_R@1"]
            ps = results[name]["ps_R@1"]
            if orig > 0:
                pct = (ps / orig - 1) * 100
                print(f"  {name:<16} {orig:.2f} -> {ps:.2f}  ({pct:+.1f}%)")

# Save
out_path = "/home/chenlibin/grepo_agent/analysis/pathswap_results.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
