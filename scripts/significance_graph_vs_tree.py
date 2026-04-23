#!/usr/bin/env python3
"""
Paired bootstrap significance test: graph-hard vs tree-neighbor negatives.
Also computes per-repo win/loss consistency.
Usage: python scripts/significance_graph_vs_tree.py
"""
import json
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASE = Path("experiments")


def load_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            key = f"{d['repo']}_{d['issue_id']}"
            preds[key] = d
    return preds


def recall_at_k(d, k=1):
    return d["metrics"][f"recall@{k}"]


graph_path = BASE / "rankft_runB_graph/eval_merged_rerank/predictions.jsonl"
tree_path = BASE / "rankft_ablation_treeneighbor/eval_merged_rerank/predictions.jsonl"

graph = load_preds(graph_path)
tree = load_preds(tree_path)

common = sorted(set(graph) & set(tree))
print(f"Common examples: {len(common)}")

for k in [1, 5, 10]:
    g_scores = [recall_at_k(graph[key], k) for key in common]
    t_scores = [recall_at_k(tree[key], k) for key in common]

    g_mean = np.mean(g_scores)
    t_mean = np.mean(t_scores)
    delta = g_mean - t_mean

    B = 10000
    n = len(common)
    boot_deltas = []
    for _ in range(B):
        idx = np.random.randint(0, n, n)
        gd = np.mean([g_scores[i] for i in idx])
        td = np.mean([t_scores[i] for i in idx])
        boot_deltas.append(gd - td)

    p_one = np.mean([d <= 0 for d in boot_deltas])
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)

    print(f"\nR@{k}: Graph={g_mean*100:.2f}%  Tree={t_mean*100:.2f}%  "
          f"Delta={delta*100:+.2f}%  p(one-sided)={p_one:.4f}  "
          f"95%CI=[{ci_lo*100:+.2f}%,{ci_hi*100:+.2f}%]")

# Per-repo consistency for R@1
print("\n--- Per-repo R@1 consistency ---")
repo_results = defaultdict(lambda: {"g": [], "t": []})
for key in common:
    repo = graph[key]["repo"]
    repo_results[repo]["g"].append(recall_at_k(graph[key], 1))
    repo_results[repo]["t"].append(recall_at_k(tree[key], 1))

wins = losses = ties = 0
win_repos = []
loss_repos = []
for repo, r in repo_results.items():
    gm, tm = np.mean(r["g"]), np.mean(r["t"])
    if gm > tm:
        wins += 1
        win_repos.append((repo, gm - tm))
    elif gm < tm:
        losses += 1
        loss_repos.append((repo, gm - tm))
    else:
        ties += 1

total = wins + losses + ties
print(f"Graph wins: {wins}/{total} repos ({wins/total*100:.0f}%)")
print(f"Graph losses: {losses}/{total} repos ({losses/total*100:.0f}%)")
print(f"Ties: {ties}/{total} repos")
print(f"\nTop repos where graph wins:")
for repo, diff in sorted(win_repos, key=lambda x: -x[1])[:5]:
    print(f"  {repo}: +{diff*100:.1f}%")
print(f"\nTop repos where tree wins:")
for repo, diff in sorted(loss_repos, key=lambda x: x[1])[:5]:
    print(f"  {repo}: {diff*100:.1f}%")
