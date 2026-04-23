#!/usr/bin/env python3
"""
Oracle Fallacy Curve: Better oracle recall can hurt end-to-end accuracy.

Key evidence for the "Oracle Fallacy" narrative in the CodeGRIP paper.
Shows that pools with higher oracle recall (gt coverage) can lead to
WORSE reranker R@1 when the reranker was trained on a different pool
distribution.

Produces:
  1. Scatter plot: Oracle Recall vs R@1 across pool types (same reranker)
  2. Per-example analysis: examples where hybrid has better oracle coverage
     but worse R@1
  3. Distribution shift heatmap: score distribution comparison

Usage:
    python scripts/oracle_fallacy_curve.py [--output_dir docs/figures]
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'blue': '#4477AA',
    'orange': '#EE6677',
    'green': '#228833',
    'purple': '#AA3377',
    'cyan': '#66CCEE',
    'gray': '#BBBBBB',
    'yellow': '#CCBB44',
}

EXP_ROOT = Path(__file__).resolve().parent.parent / "experiments"


def load_predictions(path):
    """Load predictions.jsonl and return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data


def compute_pool_metrics(data):
    """Compute oracle recall and R@1 from predictions."""
    n = len(data)
    oracle = sum(1 for d in data if d.get('gt_in_candidates', False)) / n
    r1 = sum(d['metrics']['recall@1'] for d in data) / n
    r5 = sum(d['metrics']['recall@5'] for d in data) / n
    r10 = sum(d['metrics']['recall@10'] for d in data) / n
    return {
        'oracle': oracle,
        'R@1': r1,
        'R@5': r5,
        'R@10': r10,
        'n': n,
    }


def gather_runB_graph_pools():
    """Gather all pool variants evaluated with the runB_graph reranker."""
    base = EXP_ROOT / "rankft_runB_graph"
    pool_configs = {
        'BM25 only': 'eval_bm25pool',
        'BM25 + co-change': 'eval_cochange_only',
        'BM25 + import': 'eval_import_only',
        'BM25 + both edges': 'eval_both_edge_types',
        'BM25 + graph (train dist.)': 'eval_merged_rerank',
        'BM25 + random expand': 'eval_random_expansion',
        'BM25 + E5 (hybrid)': 'eval_hybrid_only',
        'BM25 + E5 + graph': 'eval_hybrid_graph',
        'BM25 top-500': 'eval_bm25_500_rerank',
    }
    results = {}
    all_data = {}
    for label, dirname in pool_configs.items():
        pred_path = base / dirname / "predictions.jsonl"
        if pred_path.exists():
            data = load_predictions(pred_path)
            metrics = compute_pool_metrics(data)
            results[label] = metrics
            all_data[label] = data
            print(f"  {label:30s}: oracle={metrics['oracle']*100:.1f}%  "
                  f"R@1={metrics['R@1']*100:.2f}%  "
                  f"R@5={metrics['R@5']*100:.2f}%  (n={metrics['n']})")
        else:
            print(f"  {label:30s}: [not found] {pred_path}")
    return results, all_data


def gather_mixed_pool_points():
    """Gather eval points from the mixed_pool reranker (trained on BM25+graph+hybrid)."""
    base = EXP_ROOT / "rankft_mixed_pool"
    pool_configs = {
        'Mixed: BM25 only': 'eval_bm25pool',
        'Mixed: BM25 + graph': 'eval_bm25_graph',
        'Mixed: BM25 + E5 + graph': 'eval_hybrid_graph',
    }
    results = {}
    for label, dirname in pool_configs.items():
        pred_path = base / dirname / "predictions.jsonl"
        if pred_path.exists():
            data = load_predictions(pred_path)
            metrics = compute_pool_metrics(data)
            results[label] = metrics
            print(f"  {label:30s}: oracle={metrics['oracle']*100:.1f}%  "
                  f"R@1={metrics['R@1']*100:.2f}%")
    return results


def plot_oracle_vs_r1(runB_results, mixed_results, output_dir):
    """
    Figure 1: Scatter plot — Oracle Recall vs R@1.
    Shows the oracle fallacy: higher oracle does NOT imply higher R@1.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # --- RunB graph reranker (main story) ---
    # Separate train-matched pool from others
    train_label = 'BM25 + graph (train dist.)'
    xs_other, ys_other, labels_other = [], [], []
    x_train, y_train = None, None

    for label, m in runB_results.items():
        if label == train_label:
            x_train = m['oracle'] * 100
            y_train = m['R@1'] * 100
        else:
            xs_other.append(m['oracle'] * 100)
            ys_other.append(m['R@1'] * 100)
            labels_other.append(label)

    # Plot other pools
    ax.scatter(xs_other, ys_other, c=COLORS['blue'], s=60, zorder=5,
               edgecolors='white', linewidths=0.5, label='Graph reranker (OOD pool)')

    # Highlight train-matched pool
    if x_train is not None:
        ax.scatter([x_train], [y_train], c=COLORS['green'], s=120, zorder=6,
                   edgecolors='black', linewidths=1.0, marker='*',
                   label='Graph reranker (train pool)')

    # Annotate key points
    for x, y, label in zip(xs_other, ys_other, labels_other):
        short = label.replace('BM25 + ', '+').replace('BM25 ', 'BM25 ')
        # Only annotate notable ones
        if 'hybrid' in label.lower() or 'E5' in label or 'random' in label.lower():
            offset = (8, -4)
            if 'E5 + graph' in label:
                offset = (8, 4)
            ax.annotate(short, (x, y), fontsize=7.5, textcoords='offset points',
                        xytext=offset, color=COLORS['blue'], alpha=0.85)

    if x_train is not None:
        ax.annotate('Train dist.', (x_train, y_train), fontsize=8,
                    textcoords='offset points', xytext=(-10, 10),
                    color=COLORS['green'], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=0.8))

    # --- Mixed pool reranker (shows the fix) ---
    if mixed_results:
        xs_mix = [m['oracle'] * 100 for m in mixed_results.values()]
        ys_mix = [m['R@1'] * 100 for m in mixed_results.values()]
        ax.scatter(xs_mix, ys_mix, c=COLORS['orange'], s=60, zorder=5,
                   edgecolors='white', linewidths=0.5, marker='D',
                   label='Mixed reranker (pool-diverse)')

    # Draw the "oracle fallacy zone" — high oracle, low R@1
    # Shade the region where oracle > train but R@1 < train
    if x_train is not None and y_train is not None:
        ax.axhline(y=y_train, color=COLORS['gray'], linestyle=':', alpha=0.5, linewidth=0.8)
        ax.axvline(x=x_train, color=COLORS['gray'], linestyle=':', alpha=0.5, linewidth=0.8)
        # Shade: oracle > train AND R@1 < train
        ax.fill_between([x_train, 100], 0, y_train, alpha=0.07,
                         color=COLORS['orange'], zorder=0)
        ax.text(x_train + 1.5, y_train * 0.55, 'Oracle Fallacy\nZone',
                fontsize=8, fontstyle='italic', color=COLORS['orange'], alpha=0.7)

    ax.set_xlabel('Oracle Recall (% of GT files in pool)')
    ax.set_ylabel('End-to-End Recall@1 (%)')
    ax.set_title('The Oracle Fallacy: Higher Oracle $\\neq$ Higher Accuracy')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8.5)

    # Axis limits with padding
    all_x = xs_other + ([x_train] if x_train else [])
    all_y = ys_other + ([y_train] if y_train else [])
    if mixed_results:
        all_x += xs_mix
        all_y += ys_mix
    ax.set_xlim(min(all_x) - 2, max(all_x) + 3)
    ax.set_ylim(min(all_y) - 2, max(all_y) + 3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'oracle_fallacy_scatter.pdf')
    fig.savefig(path)
    path_png = os.path.join(output_dir, 'oracle_fallacy_scatter.png')
    fig.savefig(path_png)
    plt.close(fig)
    print(f"\n  Saved: {path}")
    print(f"  Saved: {path_png}")
    return path


def per_example_analysis(all_data, output_dir):
    """
    Figure 2: Per-example oracle coverage vs R@1.
    Compare graph-expanded pool (train dist.) vs hybrid pool.
    Show that examples where hybrid has BETTER oracle still have WORSE R@1.
    """
    train_key = 'BM25 + graph (train dist.)'
    hybrid_key = 'BM25 + E5 (hybrid)'

    if train_key not in all_data or hybrid_key not in all_data:
        print("  [SKIP] per-example analysis: missing data")
        return

    train_data = all_data[train_key]
    hybrid_data = all_data[hybrid_key]

    # Build per-example comparison keyed by (repo, issue_id)
    train_map = {(d['repo'], d['issue_id']): d for d in train_data}
    hybrid_map = {(d['repo'], d['issue_id']): d for d in hybrid_data}

    common_keys = set(train_map.keys()) & set(hybrid_map.keys())
    print(f"\n  Per-example analysis: {len(common_keys)} common examples")

    # Categorize examples
    categories = {
        'Both oracle hit': {'better_r1': 0, 'worse_r1': 0, 'same_r1': 0},
        'Hybrid oracle+, Train oracle-': {'better_r1': 0, 'worse_r1': 0, 'same_r1': 0},
        'Train oracle+, Hybrid oracle-': {'better_r1': 0, 'worse_r1': 0, 'same_r1': 0},
        'Both oracle miss': {'better_r1': 0, 'worse_r1': 0, 'same_r1': 0},
    }

    # Per-example score gap analysis
    r1_train_list, r1_hybrid_list = [], []
    oracle_train_list, oracle_hybrid_list = [], []
    score_gap_when_hybrid_has_oracle = []  # R@1(hybrid) - R@1(train) when hybrid has GT but train doesn't

    for key in sorted(common_keys):
        td = train_map[key]
        hd = hybrid_map[key]

        t_oracle = td['gt_in_candidates']
        h_oracle = hd['gt_in_candidates']
        t_r1 = td['metrics']['recall@1']
        h_r1 = hd['metrics']['recall@1']

        r1_train_list.append(t_r1)
        r1_hybrid_list.append(h_r1)
        oracle_train_list.append(t_oracle)
        oracle_hybrid_list.append(h_oracle)

        # Categorize
        if t_oracle and h_oracle:
            cat = 'Both oracle hit'
        elif h_oracle and not t_oracle:
            cat = 'Hybrid oracle+, Train oracle-'
            score_gap_when_hybrid_has_oracle.append(h_r1 - t_r1)
        elif t_oracle and not h_oracle:
            cat = 'Train oracle+, Hybrid oracle-'
        else:
            cat = 'Both oracle miss'

        if h_r1 > t_r1 + 1e-6:
            categories[cat]['better_r1'] += 1
        elif h_r1 < t_r1 - 1e-6:
            categories[cat]['worse_r1'] += 1
        else:
            categories[cat]['same_r1'] += 1

    print("\n  === Per-Example Oracle vs R@1 Breakdown ===")
    print(f"  {'Category':42s} {'Hybrid R@1 better':>18s} {'Same':>8s} {'Hybrid R@1 worse':>18s}")
    print("  " + "-" * 90)
    for cat, counts in categories.items():
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"  {cat:42s} {counts['better_r1']:>8d} ({counts['better_r1']/total*100:5.1f}%)"
              f" {counts['same_r1']:>6d}"
              f" {counts['worse_r1']:>8d} ({counts['worse_r1']/total*100:5.1f}%)")

    # Key statistic: among examples where ONLY hybrid has oracle coverage
    if score_gap_when_hybrid_has_oracle:
        gaps = np.array(score_gap_when_hybrid_has_oracle)
        n_better = np.sum(gaps > 1e-6)
        n_worse = np.sum(gaps < -1e-6)
        n_same = len(gaps) - n_better - n_worse
        print(f"\n  When hybrid has oracle coverage but train pool does NOT ({len(gaps)} examples):")
        print(f"    Hybrid R@1 better: {n_better} ({n_better/len(gaps)*100:.1f}%)")
        print(f"    Hybrid R@1 worse:  {n_worse} ({n_worse/len(gaps)*100:.1f}%)")
        print(f"    Same:              {n_same} ({n_same/len(gaps)*100:.1f}%)")
        print(f"    Mean R@1 gap:      {gaps.mean()*100:.2f}pp")

    # --- Figure 2: Stacked bar showing the breakdown ---
    fig, ax = plt.subplots(figsize=(6, 3.5))

    cat_labels = [c for c in categories if sum(categories[c].values()) > 0]
    better = [categories[c]['better_r1'] for c in cat_labels]
    same = [categories[c]['same_r1'] for c in cat_labels]
    worse = [categories[c]['worse_r1'] for c in cat_labels]
    totals = [b + s + w for b, s, w in zip(better, same, worse)]

    # Normalize to percentages
    better_pct = [b / t * 100 if t > 0 else 0 for b, t in zip(better, totals)]
    same_pct = [s / t * 100 if t > 0 else 0 for s, t in zip(same, totals)]
    worse_pct = [w / t * 100 if t > 0 else 0 for w, t in zip(worse, totals)]

    y_pos = np.arange(len(cat_labels))
    bar_h = 0.5

    # Shorten labels
    short_labels = []
    for c in cat_labels:
        c = c.replace('Hybrid oracle+, Train oracle-', 'Only hybrid has GT')
        c = c.replace('Train oracle+, Hybrid oracle-', 'Only train pool has GT')
        short_labels.append(c)

    bars1 = ax.barh(y_pos, better_pct, bar_h, color=COLORS['green'], label='Hybrid R@1 better')
    bars2 = ax.barh(y_pos, same_pct, bar_h, left=better_pct, color=COLORS['gray'], label='Same')
    left2 = [b + s for b, s in zip(better_pct, same_pct)]
    bars3 = ax.barh(y_pos, worse_pct, bar_h, left=left2, color=COLORS['orange'], label='Hybrid R@1 worse')

    # Add count annotations
    for i, (b, s, w, t) in enumerate(zip(better, same, worse, totals)):
        ax.text(101, i, f'n={t}', va='center', fontsize=8, color='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel('Percentage of Examples (%)')
    ax.set_title('Per-Example: Oracle Coverage vs Actual R@1\n(Hybrid pool vs Graph-expanded pool, same reranker)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, 'oracle_fallacy_per_example.pdf')
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, 'oracle_fallacy_per_example.png'))
    plt.close(fig)
    print(f"\n  Saved: {path}")


def score_distribution_analysis(all_data, output_dir):
    """
    Figure 3: Score distribution comparison.
    Shows that on the hybrid pool, the reranker's scores are less calibrated
    — it assigns high scores to wrong candidates.
    """
    train_key = 'BM25 + graph (train dist.)'
    hybrid_key = 'BM25 + E5 (hybrid)'

    if train_key not in all_data or hybrid_key not in all_data:
        print("  [SKIP] score distribution analysis: missing data")
        return

    train_data = all_data[train_key]
    hybrid_data = all_data[hybrid_key]

    # Collect top-1 scores and whether top-1 is correct
    train_scores_correct, train_scores_wrong = [], []
    hybrid_scores_correct, hybrid_scores_wrong = [], []

    for d in train_data:
        if len(d['scores']) == 0:
            continue
        top_score = d['scores'][0]
        is_correct = d['metrics']['recall@1'] > 0.5  # at least one GT in top-1
        if is_correct:
            train_scores_correct.append(top_score)
        else:
            train_scores_wrong.append(top_score)

    for d in hybrid_data:
        if len(d['scores']) == 0:
            continue
        top_score = d['scores'][0]
        is_correct = d['metrics']['recall@1'] > 0.5
        if is_correct:
            hybrid_scores_correct.append(top_score)
        else:
            hybrid_scores_wrong.append(top_score)

    print(f"\n  Score distribution:")
    print(f"    Train pool - correct top-1: n={len(train_scores_correct)}, "
          f"mean={np.mean(train_scores_correct):.2f}")
    print(f"    Train pool - wrong   top-1: n={len(train_scores_wrong)}, "
          f"mean={np.mean(train_scores_wrong):.2f}")
    print(f"    Hybrid pool - correct top-1: n={len(hybrid_scores_correct)}, "
          f"mean={np.mean(hybrid_scores_correct):.2f}")
    print(f"    Hybrid pool - wrong   top-1: n={len(hybrid_scores_wrong)}, "
          f"mean={np.mean(hybrid_scores_wrong):.2f}")

    # Score margin analysis: gap between top-1 and top-2
    train_margins_correct, train_margins_wrong = [], []
    hybrid_margins_correct, hybrid_margins_wrong = [], []

    for d in train_data:
        if len(d['scores']) < 2:
            continue
        margin = d['scores'][0] - d['scores'][1]
        is_correct = d['metrics']['recall@1'] > 0.5
        if is_correct:
            train_margins_correct.append(margin)
        else:
            train_margins_wrong.append(margin)

    for d in hybrid_data:
        if len(d['scores']) < 2:
            continue
        margin = d['scores'][0] - d['scores'][1]
        is_correct = d['metrics']['recall@1'] > 0.5
        if is_correct:
            hybrid_margins_correct.append(margin)
        else:
            hybrid_margins_wrong.append(margin)

    # --- Figure 3: Score distribution violin/histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=False)

    # Panel A: Top-1 score distributions
    ax = axes[0]
    all_scores = (train_scores_correct + train_scores_wrong +
                  hybrid_scores_correct + hybrid_scores_wrong)
    bins = np.linspace(min(all_scores) - 0.5, max(all_scores) + 0.5, 40)

    ax.hist(train_scores_correct, bins=bins, alpha=0.5, color=COLORS['green'],
            label=f'Train pool, correct (n={len(train_scores_correct)})', density=True)
    ax.hist(train_scores_wrong, bins=bins, alpha=0.5, color=COLORS['blue'],
            label=f'Train pool, wrong (n={len(train_scores_wrong)})', density=True)
    ax.hist(hybrid_scores_wrong, bins=bins, alpha=0.3, color=COLORS['orange'],
            label=f'Hybrid pool, wrong (n={len(hybrid_scores_wrong)})',
            density=True, histtype='step', linewidth=1.5)

    ax.set_xlabel('Top-1 Reranker Score')
    ax.set_ylabel('Density')
    ax.set_title('(a) Top-1 Score Distribution')
    ax.legend(fontsize=7, loc='upper left')

    # Panel B: Score margin (top1 - top2)
    ax = axes[1]
    all_margins = (train_margins_correct + train_margins_wrong +
                   hybrid_margins_correct + hybrid_margins_wrong)
    bins_m = np.linspace(min(all_margins) - 0.1, max(all_margins) + 0.1, 40)

    ax.hist(train_margins_correct, bins=bins_m, alpha=0.5, color=COLORS['green'],
            label=f'Train pool, correct', density=True)
    ax.hist(hybrid_margins_wrong, bins=bins_m, alpha=0.4, color=COLORS['orange'],
            label=f'Hybrid pool, wrong', density=True)

    ax.set_xlabel('Score Margin (top-1 $-$ top-2)')
    ax.set_ylabel('Density')
    ax.set_title('(b) Confidence Margin')
    ax.legend(fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(output_dir, 'oracle_fallacy_scores.pdf')
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, 'oracle_fallacy_scores.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


def contamination_curve(all_data, output_dir):
    """
    Figure 4: Pool contamination curve.
    Simulate gradual replacement of train-distribution candidates with
    hybrid-pool candidates. At each contamination level, measure what
    fraction of examples would have their top-1 prediction changed.

    This is a proxy analysis: we identify candidates unique to each pool
    and measure how the top-ranked candidate stability degrades as the
    pool composition shifts.
    """
    train_key = 'BM25 + graph (train dist.)'
    hybrid_key = 'BM25 + E5 (hybrid)'

    if train_key not in all_data or hybrid_key not in all_data:
        print("  [SKIP] contamination curve: missing data")
        return

    train_data = all_data[train_key]
    hybrid_data = all_data[hybrid_key]

    train_map = {(d['repo'], d['issue_id']): d for d in train_data}
    hybrid_map = {(d['repo'], d['issue_id']): d for d in hybrid_data}
    common_keys = sorted(set(train_map.keys()) & set(hybrid_map.keys()))

    # For each example, compare candidate pools
    # predicted list = reranked candidates (top 50)
    # We analyze: what fraction of train pool's top-K are absent from hybrid pool?

    contamination_levels = np.arange(0, 1.01, 0.05)
    r1_at_level = []

    # Strategy: At contamination level p, for each example:
    # - With prob p, use the hybrid pool's top-1 prediction
    # - With prob (1-p), use the train pool's top-1 prediction
    # This simulates the expected R@1 of a mixed pool at the example level.
    #
    # More precisely: we have the train pool's ranked list and hybrid pool's
    # ranked list. At level p, we model the contaminated pool as having
    # the top-1 from train with prob (1-p) and from hybrid with prob p.
    # The expected R@1 = (1-p) * R@1_train + p * R@1_hybrid

    # But that's just linear interpolation. We want something more insightful.
    # Instead: measure per-example the overlap in top-K candidates and how
    # contamination disrupts the reranker's ability to find GT.

    # Approach: For each example, look at how many of the train pool's
    # top-K candidates are shared with the hybrid pool. When they differ,
    # the reranker sees unfamiliar candidates from the hybrid distribution.

    # Candidate overlap analysis
    top_k_values = [1, 3, 5, 10, 20, 50]
    overlap_stats = {k: [] for k in top_k_values}

    for key in common_keys:
        td = train_map[key]
        hd = hybrid_map[key]
        t_pred = td['predicted']
        h_pred = hd['predicted']

        for k in top_k_values:
            t_set = set(t_pred[:k])
            h_set = set(h_pred[:k])
            if len(t_set) > 0:
                overlap = len(t_set & h_set) / k
                overlap_stats[k].append(overlap)

    print(f"\n  === Candidate Overlap (Train pool top-K vs Hybrid pool top-K) ===")
    for k in top_k_values:
        vals = overlap_stats[k]
        print(f"    Top-{k:2d}: overlap = {np.mean(vals)*100:.1f}% "
              f"(std={np.std(vals)*100:.1f}%)")

    # --- Contamination simulation ---
    # For each example, we have train_predicted and hybrid_predicted (both scored).
    # At contamination level p:
    #   - Take (1-p) fraction of candidates from train pool
    #   - Take p fraction from hybrid pool
    #   - The reranker's ranking within train candidates is valid (in-dist)
    #   - The reranker's ranking within hybrid candidates may be unreliable
    # Proxy: at level p, take top round((1-p)*50) from train pool +
    #        top round(p*50) from hybrid pool, then check if GT is in top-1
    #        of the train portion (since reranker is calibrated there).

    contamination_levels = np.linspace(0, 1, 21)
    mean_r1 = []
    mean_oracle = []

    for p in contamination_levels:
        n_train = max(1, int(round((1 - p) * 50)))
        n_hybrid = 50 - n_train

        r1_hits = 0
        oracle_hits = 0
        n_total = 0

        for key in common_keys:
            td = train_map[key]
            hd = hybrid_map[key]

            gt = set(td['ground_truth'])
            t_pred = td['predicted'][:n_train]
            h_pred = hd['predicted'][:n_hybrid]

            # Pool = union of train top-n_train and hybrid top-n_hybrid
            # but ranked: train candidates first (in-distribution), then hybrid
            combined = list(t_pred)
            seen = set(t_pred)
            for f in h_pred:
                if f not in seen:
                    combined.append(f)
                    seen.add(f)

            # Oracle: is any GT file in the combined pool?
            pool_set = set(combined)
            if pool_set & gt:
                oracle_hits += 1

            # R@1: is the top-1 of combined (= train's top-1 if n_train >= 1) in GT?
            if len(combined) > 0 and combined[0] in gt:
                r1_hits += 1

            n_total += 1

        mean_r1.append(r1_hits / n_total * 100)
        mean_oracle.append(oracle_hits / n_total * 100)

    # Also compute a "hybrid-first" variant where hybrid candidates are prioritized
    mean_r1_hybrid_first = []
    mean_oracle_hybrid_first = []

    for p in contamination_levels:
        n_hybrid = max(1, int(round(p * 50)))
        n_train = 50 - n_hybrid

        r1_hits = 0
        oracle_hits = 0
        n_total = 0

        for key in common_keys:
            td = train_map[key]
            hd = hybrid_map[key]

            gt = set(td['ground_truth'])
            h_pred = hd['predicted'][:n_hybrid]
            t_pred = td['predicted'][:n_train]

            # Hybrid first, then train
            combined = list(h_pred)
            seen = set(h_pred)
            for f in t_pred:
                if f not in seen:
                    combined.append(f)
                    seen.add(f)

            pool_set = set(combined)
            if pool_set & gt:
                oracle_hits += 1

            if len(combined) > 0 and combined[0] in gt:
                r1_hits += 1

            n_total += 1

        mean_r1_hybrid_first.append(r1_hits / n_total * 100)
        mean_oracle_hybrid_first.append(oracle_hits / n_total * 100)

    # --- Figure 4: Contamination curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    pcts = contamination_levels * 100

    # Panel A: Oracle recall vs contamination
    ax1.plot(pcts, mean_oracle, '-o', color=COLORS['blue'], markersize=3,
             label='Oracle Recall', linewidth=1.5)
    ax1.plot(pcts, mean_r1, '-s', color=COLORS['orange'], markersize=3,
             label='R@1 (train pool ranked first)', linewidth=1.5)
    ax1.plot(pcts, mean_r1_hybrid_first, '--^', color=COLORS['purple'],
             markersize=3, label='R@1 (hybrid pool ranked first)', linewidth=1.5)

    ax1.set_xlabel('Contamination Level (% hybrid candidates)')
    ax1.set_ylabel('Metric (%)')
    ax1.set_title('(a) Pool Contamination Curve')
    ax1.legend(fontsize=7.5, loc='center right')

    # Panel B: The gap — Oracle minus R@1
    gap_train_first = np.array(mean_oracle) - np.array(mean_r1)
    gap_hybrid_first = np.array(mean_oracle_hybrid_first) - np.array(mean_r1_hybrid_first)

    ax2.plot(pcts, gap_train_first, '-o', color=COLORS['blue'], markersize=3,
             label='Oracle - R@1 (train first)', linewidth=1.5)
    ax2.plot(pcts, gap_hybrid_first, '--^', color=COLORS['purple'], markersize=3,
             label='Oracle - R@1 (hybrid first)', linewidth=1.5)
    ax2.fill_between(pcts, gap_train_first, alpha=0.15, color=COLORS['blue'])

    ax2.set_xlabel('Contamination Level (% hybrid candidates)')
    ax2.set_ylabel('Oracle - R@1 Gap (pp)')
    ax2.set_title('(b) Growing Oracle-Accuracy Gap')
    ax2.legend(fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(output_dir, 'oracle_fallacy_contamination.pdf')
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, 'oracle_fallacy_contamination.png'))
    plt.close(fig)
    print(f"  Saved: {path}")


def summary_table(runB_results, mixed_results):
    """Print a summary table of all data points for the paper."""
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE: Oracle Fallacy Evidence")
    print("=" * 80)
    print(f"  {'Reranker':20s} {'Pool':30s} {'Oracle':>8s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s}")
    print("  " + "-" * 78)

    for label, m in sorted(runB_results.items(), key=lambda x: x[1]['oracle']):
        print(f"  {'Graph reranker':20s} {label:30s} "
              f"{m['oracle']*100:7.1f}% {m['R@1']*100:7.2f}% "
              f"{m['R@5']*100:7.2f}% {m['R@10']*100:7.2f}%")

    if mixed_results:
        print("  " + "-" * 78)
        for label, m in sorted(mixed_results.items(), key=lambda x: x[1]['oracle']):
            print(f"  {'Mixed reranker':20s} {label:30s} "
                  f"{m['oracle']*100:7.1f}% {m['R@1']*100:7.2f}% "
                  f"{m['R@5']*100:7.2f}% {m['R@10']*100:7.2f}%")

    print("=" * 80)

    # Highlight the key finding
    train_m = runB_results.get('BM25 + graph (train dist.)')
    hybrid_m = runB_results.get('BM25 + E5 (hybrid)')
    if train_m and hybrid_m:
        print(f"\n  KEY FINDING:")
        print(f"    Hybrid pool oracle:  {hybrid_m['oracle']*100:.1f}% "
              f"(+{(hybrid_m['oracle']-train_m['oracle'])*100:.1f}pp vs train pool)")
        print(f"    Hybrid pool R@1:     {hybrid_m['R@1']*100:.2f}% "
              f"({(hybrid_m['R@1']-train_m['R@1'])*100:+.2f}pp vs train pool)")
        print(f"    => Higher oracle recall ({hybrid_m['oracle']*100:.1f}% > {train_m['oracle']*100:.1f}%) "
              f"but LOWER R@1 ({hybrid_m['R@1']*100:.2f}% < {train_m['R@1']*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Oracle Fallacy Curve for CodeGRIP")
    parser.add_argument('--output_dir', default='docs/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    # Resolve relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Gathering RunB Graph Reranker Results ===")
    runB_results, all_data = gather_runB_graph_pools()

    print("\n=== Gathering Mixed Pool Reranker Results ===")
    mixed_results = gather_mixed_pool_points()

    print("\n=== Figure 1: Oracle vs R@1 Scatter ===")
    plot_oracle_vs_r1(runB_results, mixed_results, str(output_dir))

    print("\n=== Figure 2: Per-Example Analysis ===")
    per_example_analysis(all_data, str(output_dir))

    print("\n=== Figure 3: Score Distribution Analysis ===")
    score_distribution_analysis(all_data, str(output_dir))

    print("\n=== Figure 4: Pool Contamination Curve ===")
    contamination_curve(all_data, str(output_dir))

    summary_table(runB_results, mixed_results)


if __name__ == '__main__':
    main()
