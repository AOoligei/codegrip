#!/usr/bin/env python3
"""
Detailed analysis of model predictions from eval_merged_rerank.

Analyzes:
1. Score distributions (GT vs non-GT) + AUC
2. Error analysis by difficulty (num GT files, repo size, BM25 overlap)
3. Path pattern analysis (edit distance, directory overlap)
4. Confidence calibration

Outputs text summary to stdout and figures to experiments/analysis/.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments/rankft_runB_graph/eval_merged_rerank/predictions.jsonl",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments/analysis",
)
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_predictions(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# 1. Score Distribution Analysis
# ---------------------------------------------------------------------------
def score_distribution_analysis(data: list[dict]):
    gt_scores = []
    non_gt_scores = []
    all_labels = []
    all_scores = []

    for sample in data:
        gt_set = set(sample["ground_truth"])
        predicted = sample["predicted"]
        scores = sample["scores"]
        for pred_file, score in zip(predicted, scores):
            if pred_file in gt_set:
                gt_scores.append(score)
                all_labels.append(1)
            else:
                non_gt_scores.append(score)
                all_labels.append(0)
            all_scores.append(score)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # AUC
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(all_labels, all_scores)

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    # --- Plot histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(min(all_scores), max(all_scores), 80)
    ax.hist(gt_scores, bins=bins, alpha=0.6, label=f"GT files (n={len(gt_scores)})", density=True)
    ax.hist(non_gt_scores, bins=bins, alpha=0.6, label=f"Non-GT files (n={len(non_gt_scores)})", density=True)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: GT vs Non-GT")
    ax.legend()
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    # --- Plot ROC ---
    ax = axes[1]
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=150)
    plt.close(fig)

    # Stats
    print("=" * 70)
    print("1. SCORE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Total file-level predictions: {len(all_scores)}")
    print(f"  GT file scores:     n={len(gt_scores):>6d}  mean={np.mean(gt_scores):.3f}  median={np.median(gt_scores):.3f}  std={np.std(gt_scores):.3f}")
    print(f"  Non-GT file scores: n={len(non_gt_scores):>6d}  mean={np.mean(non_gt_scores):.3f}  median={np.median(non_gt_scores):.3f}  std={np.std(non_gt_scores):.3f}")
    print(f"  AUC (ROC): {auc:.4f}")
    print(f"  Score separation (mean diff): {np.mean(gt_scores) - np.mean(non_gt_scores):.3f}")
    print()

    return auc


# ---------------------------------------------------------------------------
# 2. Error Analysis by Difficulty
# ---------------------------------------------------------------------------
def error_analysis_by_difficulty(data: list[dict]):
    print("=" * 70)
    print("2. ERROR ANALYSIS BY DIFFICULTY")
    print("=" * 70)

    # --- 2a. Group by number of GT files ---
    gt_groups = {"1": [], "2": [], "3+": []}
    for sample in data:
        n_gt = len(sample["ground_truth"])
        if n_gt == 1:
            gt_groups["1"].append(sample)
        elif n_gt == 2:
            gt_groups["2"].append(sample)
        else:
            gt_groups["3+"].append(sample)

    print("\n  --- By Number of GT Files ---")
    print(f"  {'Group':<10} {'N':>6} {'Hit@1':>8} {'Acc@1':>8} {'Recall@5':>10} {'Recall@10':>10}")
    metrics_by_gt = {}
    for group_name, samples in gt_groups.items():
        h1 = np.mean([s["metrics"]["hit@1"] for s in samples])
        a1 = np.mean([s["metrics"]["acc@1"] for s in samples])
        r5 = np.mean([s["metrics"]["recall@5"] for s in samples])
        r10 = np.mean([s["metrics"]["recall@10"] for s in samples])
        print(f"  {group_name:<10} {len(samples):>6} {h1:>8.4f} {a1:>8.4f} {r5:>10.4f} {r10:>10.4f}")
        metrics_by_gt[group_name] = {"hit@1": h1, "acc@1": a1, "recall@5": r5, "recall@10": r10, "n": len(samples)}

    # --- 2b. Group by repo size (num_candidates as proxy) ---
    size_groups = {"small (<100)": [], "medium (100-150)": [], "large (150+)": []}
    for sample in data:
        nc = sample["num_candidates"]
        if nc < 100:
            size_groups["small (<100)"].append(sample)
        elif nc < 150:
            size_groups["medium (100-150)"].append(sample)
        else:
            size_groups["large (150+)"].append(sample)

    print("\n  --- By Repo Size (num_candidates as proxy) ---")
    print(f"  {'Group':<20} {'N':>6} {'Hit@1':>8} {'Acc@1':>8} {'Recall@5':>10} {'Recall@10':>10}")
    for group_name, samples in size_groups.items():
        if not samples:
            continue
        h1 = np.mean([s["metrics"]["hit@1"] for s in samples])
        a1 = np.mean([s["metrics"]["acc@1"] for s in samples])
        r5 = np.mean([s["metrics"]["recall@5"] for s in samples])
        r10 = np.mean([s["metrics"]["recall@10"] for s in samples])
        print(f"  {group_name:<20} {len(samples):>6} {h1:>8.4f} {a1:>8.4f} {r5:>10.4f} {r10:>10.4f}")

    # --- 2c. Group by whether GT file is in BM25 top-10 ---
    bm25_in = []
    bm25_out = []
    for sample in data:
        gt_set = set(sample["ground_truth"])
        bm25_top10 = set(sample["bm25_original"][:10])
        if gt_set & bm25_top10:
            bm25_in.append(sample)
        else:
            bm25_out.append(sample)

    print("\n  --- By GT in BM25 Top-10 ---")
    print(f"  {'Group':<25} {'N':>6} {'Hit@1':>8} {'Acc@1':>8} {'Recall@5':>10} {'Recall@10':>10}")
    for label, samples in [("GT in BM25 top-10", bm25_in), ("GT NOT in BM25 top-10", bm25_out)]:
        if not samples:
            continue
        h1 = np.mean([s["metrics"]["hit@1"] for s in samples])
        a1 = np.mean([s["metrics"]["acc@1"] for s in samples])
        r5 = np.mean([s["metrics"]["recall@5"] for s in samples])
        r10 = np.mean([s["metrics"]["recall@10"] for s in samples])
        print(f"  {label:<25} {len(samples):>6} {h1:>8.4f} {a1:>8.4f} {r5:>10.4f} {r10:>10.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # By GT count
    ax = axes[0]
    groups = list(gt_groups.keys())
    hit1_vals = [metrics_by_gt[g]["hit@1"] for g in groups]
    r5_vals = [metrics_by_gt[g]["recall@5"] for g in groups]
    r10_vals = [metrics_by_gt[g]["recall@10"] for g in groups]
    x = np.arange(len(groups))
    w = 0.25
    ax.bar(x - w, hit1_vals, w, label="Hit@1")
    ax.bar(x, r5_vals, w, label="Recall@5")
    ax.bar(x + w, r10_vals, w, label="Recall@10")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={metrics_by_gt[g]['n']})" for g in groups])
    ax.set_ylabel("Score")
    ax.set_title("Performance by # GT Files")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # By repo size
    ax = axes[1]
    size_names = []
    size_h1 = []
    size_r5 = []
    for gn, ss in size_groups.items():
        if ss:
            size_names.append(f"{gn}\n(n={len(ss)})")
            size_h1.append(np.mean([s["metrics"]["hit@1"] for s in ss]))
            size_r5.append(np.mean([s["metrics"]["recall@5"] for s in ss]))
    x = np.arange(len(size_names))
    ax.bar(x - 0.15, size_h1, 0.3, label="Hit@1")
    ax.bar(x + 0.15, size_r5, 0.3, label="Recall@5")
    ax.set_xticks(x)
    ax.set_xticklabels(size_names, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Performance by Repo Size")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # By BM25
    ax = axes[2]
    bm_labels = ["GT in BM25\ntop-10", "GT NOT in\nBM25 top-10"]
    bm_h1 = []
    bm_r5 = []
    bm_ns = []
    for ss in [bm25_in, bm25_out]:
        if ss:
            bm_h1.append(np.mean([s["metrics"]["hit@1"] for s in ss]))
            bm_r5.append(np.mean([s["metrics"]["recall@5"] for s in ss]))
            bm_ns.append(len(ss))
    x = np.arange(len(bm_h1))
    ax.bar(x - 0.15, bm_h1, 0.3, label="Hit@1")
    ax.bar(x + 0.15, bm_r5, 0.3, label="Recall@5")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(bm_labels, bm_ns)], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Performance by BM25 Overlap")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "error_by_difficulty.png"), dpi=150)
    plt.close(fig)
    print()


# ---------------------------------------------------------------------------
# 3. Path Pattern Analysis
# ---------------------------------------------------------------------------
def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance on path components."""
    p1 = s1.split("/")
    p2 = s2.split("/")
    n, m = len(p1), len(p2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if p1[i - 1] == p2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def _same_dir(a: str, b: str) -> bool:
    return os.path.dirname(a) == os.path.dirname(b)


def path_pattern_analysis(data: list[dict]):
    print("=" * 70)
    print("3. PATH PATTERN ANALYSIS")
    print("=" * 70)

    # For each sample, look at top-1 prediction vs closest GT file
    correct_dirs = defaultdict(int)    # directory of correctly predicted files
    incorrect_dirs = defaultdict(int)  # directory of incorrectly predicted files

    edit_distances = []         # edit distance of top-1 to nearest GT
    same_dir_when_wrong = 0     # wrong top-1 but same directory as a GT
    total_wrong = 0
    same_dir_when_correct = 0
    total_correct = 0

    # Extension analysis
    correct_ext_match = 0
    wrong_ext_match = 0

    for sample in data:
        gt_set = set(sample["ground_truth"])
        predicted = sample["predicted"]
        if not predicted:
            continue

        top1 = predicted[0]
        is_correct = top1 in gt_set

        # Edit distance to nearest GT
        min_ed = min(_edit_distance(top1, gt) for gt in gt_set)
        edit_distances.append(min_ed)

        if is_correct:
            total_correct += 1
            d = os.path.dirname(top1)
            correct_dirs[d] += 1
        else:
            total_wrong += 1
            d = os.path.dirname(top1)
            incorrect_dirs[d] += 1
            # Is wrong prediction in same dir as any GT?
            if any(_same_dir(top1, gt) for gt in gt_set):
                same_dir_when_wrong += 1
            # Extension match
            top1_ext = os.path.splitext(top1)[1]
            if any(os.path.splitext(gt)[1] == top1_ext for gt in gt_set):
                wrong_ext_match += 1

    edit_distances = np.array(edit_distances)

    print(f"\n  Top-1 correct: {total_correct}/{len(data)} ({total_correct/len(data)*100:.1f}%)")
    print(f"  Top-1 wrong:   {total_wrong}/{len(data)} ({total_wrong/len(data)*100:.1f}%)")
    print(f"\n  When wrong:")
    print(f"    Same directory as a GT file: {same_dir_when_wrong}/{total_wrong} ({same_dir_when_wrong/max(total_wrong,1)*100:.1f}%)")
    print(f"\n  Edit distance (path components) from top-1 to nearest GT:")
    print(f"    Mean: {edit_distances.mean():.2f}  Median: {np.median(edit_distances):.2f}  Std: {edit_distances.std():.2f}")
    print(f"    Distribution: 0={np.sum(edit_distances==0)}  1={np.sum(edit_distances==1)}  2={np.sum(edit_distances==2)}  3+={np.sum(edit_distances>=3)}")

    # Top incorrect directories
    print(f"\n  Top-10 directories with most WRONG top-1 predictions:")
    for d, cnt in sorted(incorrect_dirs.items(), key=lambda x: -x[1])[:10]:
        print(f"    {d or '(root)':<60s} {cnt}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    max_ed = min(int(edit_distances.max()), 10)
    bins = np.arange(0, max_ed + 2) - 0.5
    ax.hist(edit_distances[edit_distances <= max_ed], bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Edit Distance (path components)")
    ax.set_ylabel("Count")
    ax.set_title("Top-1 to Nearest GT: Path Edit Distance")
    ax.set_xticks(range(max_ed + 1))

    # Same-dir analysis for wrong predictions at different ranks
    ax = axes[1]
    ranks_to_check = [1, 3, 5, 10, 20]
    same_dir_rates = []
    for k in ranks_to_check:
        cnt = 0
        total = 0
        for sample in data:
            gt_set = set(sample["ground_truth"])
            top_k = sample["predicted"][:k]
            # Among top-k that are wrong, how many share dir with GT?
            for p in top_k:
                if p not in gt_set:
                    total += 1
                    if any(_same_dir(p, gt) for gt in gt_set):
                        cnt += 1
        same_dir_rates.append(cnt / max(total, 1))
    ax.bar(range(len(ranks_to_check)), same_dir_rates, tick_label=[f"Top-{k}" for k in ranks_to_check],
           alpha=0.7, edgecolor="black")
    ax.set_ylabel("Fraction")
    ax.set_title("Wrong Predictions Sharing Directory with GT")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "path_patterns.png"), dpi=150)
    plt.close(fig)
    print()


# ---------------------------------------------------------------------------
# 4. Confidence Calibration
# ---------------------------------------------------------------------------
def confidence_calibration(data: list[dict]):
    print("=" * 70)
    print("4. CONFIDENCE CALIBRATION")
    print("=" * 70)

    top1_scores = []
    top1_correct = []

    for sample in data:
        gt_set = set(sample["ground_truth"])
        if not sample["predicted"]:
            continue
        top1 = sample["predicted"][0]
        score = sample["scores"][0]
        top1_scores.append(score)
        top1_correct.append(1 if top1 in gt_set else 0)

    top1_scores = np.array(top1_scores)
    top1_correct = np.array(top1_correct)

    # Bin by score percentiles (10 bins)
    n_bins = 10
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(top1_scores, percentiles)
    # Make edges unique
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1

    bin_centers = []
    bin_accs = []
    bin_counts = []
    bin_labels = []

    print(f"\n  {'Score Range':<25s} {'N':>6} {'Accuracy':>10} {'Mean Score':>12}")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (top1_scores >= lo) & (top1_scores < hi)
        else:
            mask = (top1_scores >= lo) & (top1_scores <= hi)
        if mask.sum() == 0:
            continue
        acc = top1_correct[mask].mean()
        mean_score = top1_scores[mask].mean()
        n = mask.sum()
        bin_centers.append(mean_score)
        bin_accs.append(acc)
        bin_counts.append(n)
        label = f"[{lo:.2f}, {hi:.2f})"
        bin_labels.append(label)
        print(f"  {label:<25s} {n:>6} {acc:>10.4f} {mean_score:>12.3f}")

    # Overall
    print(f"\n  Overall top-1 accuracy (hit@1): {top1_correct.mean():.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(len(bin_accs)), bin_accs, tick_label=[f"{c:.1f}" for c in bin_centers],
           alpha=0.7, edgecolor="black")
    ax.set_xlabel("Mean Score in Bin")
    ax.set_ylabel("Accuracy (Hit@1)")
    ax.set_title("Calibration: Accuracy by Score Bin")
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax = axes[1]
    ax.bar(range(len(bin_counts)), bin_counts, tick_label=[f"{c:.1f}" for c in bin_centers],
           alpha=0.7, edgecolor="black", color="orange")
    ax.set_xlabel("Mean Score in Bin")
    ax.set_ylabel("Count")
    ax.set_title("Sample Count per Score Bin")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration.png"), dpi=150)
    plt.close(fig)

    # Reliability diagram style: score vs accuracy correlation
    from scipy.stats import spearmanr
    if len(bin_accs) > 2:
        rho, p = spearmanr(bin_centers, bin_accs)
        print(f"  Spearman correlation (score vs accuracy): rho={rho:.4f}, p={p:.4f}")

    print()


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
def print_summary(data: list[dict], auc: float):
    print("=" * 70)
    print("5. OVERALL SUMMARY")
    print("=" * 70)

    # Aggregate metrics
    metric_keys = ["hit@1", "acc@1", "recall@1", "hit@5", "acc@5", "recall@5",
                   "hit@10", "acc@10", "recall@10", "hit@20", "acc@20", "recall@20"]
    print(f"\n  {'Metric':<15s} {'Mean':>8}")
    for mk in metric_keys:
        vals = [s["metrics"][mk] for s in data]
        print(f"  {mk:<15s} {np.mean(vals):>8.4f}")

    print(f"\n  AUC (file-level GT vs non-GT separation): {auc:.4f}")
    print(f"  Total samples: {len(data)}")
    print(f"  Unique repos: {len(set(s['repo'] for s in data))}")

    # Per-repo hit@1
    repo_metrics = defaultdict(list)
    for s in data:
        repo_metrics[s["repo"]].append(s["metrics"]["hit@1"])

    print(f"\n  Per-repo Hit@1 (top-5 / bottom-5):")
    repo_avgs = {r: np.mean(v) for r, v in repo_metrics.items()}
    sorted_repos = sorted(repo_avgs.items(), key=lambda x: -x[1])
    print(f"    {'Repo':<35s} {'N':>5} {'Hit@1':>8}")
    for r, v in sorted_repos[:5]:
        print(f"    {r:<35s} {len(repo_metrics[r]):>5} {v:>8.4f}")
    print(f"    {'...'}")
    for r, v in sorted_repos[-5:]:
        print(f"    {r:<35s} {len(repo_metrics[r]):>5} {v:>8.4f}")

    print(f"\n  Figures saved to: {OUT_DIR}/")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading predictions from: {PRED_PATH}")
    data = load_predictions(PRED_PATH)
    print(f"Loaded {len(data)} samples\n")

    auc = score_distribution_analysis(data)
    error_analysis_by_difficulty(data)
    path_pattern_analysis(data)
    confidence_calibration(data)
    print_summary(data, auc)


if __name__ == "__main__":
    main()
