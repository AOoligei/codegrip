#!/usr/bin/env python3
"""
Generate publication-quality figures for the CodeGRIP paper.

Figures:
1. Oracle Gap Bar Chart: actual vs oracle across pipeline stages
2. Reranker Feature Importance: horizontal bar chart
3. Bottleneck Decomposition: stacked/pie chart of where GT files are lost
4. Expansion Signal Ablation: grouped bar chart
5. Hit@K Comparison: multi-bar chart across methods

Usage:
    python scripts/generate_figures.py --output_dir docs/figures
"""

import json
import os
import argparse

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

# Color palette (colorblind-friendly)
COLORS = {
    'blue': '#4477AA',
    'orange': '#EE6677',
    'green': '#228833',
    'purple': '#AA3377',
    'cyan': '#66CCEE',
    'gray': '#BBBBBB',
    'dark': '#333333',
}


def fig1_oracle_gap(output_dir):
    """Oracle Gap: actual vs oracle performance across pipeline stages."""
    stages = ['Base\n(SFT)', 'Expanded', 'Reranked\n(rule)', 'Reranked\n(learned)']
    ks = [5, 10, 20]

    actual = {
        5:  [25.38, 31.13, 33.24, 34.66],
        10: [26.54, 36.34, 38.57, 40.65],
        20: [27.13, 41.27, 42.79, 45.81],
    }
    oracle = {
        5:  [27.06, 47.53, 47.53, 47.53],
        10: [27.11, 48.13, 48.13, 48.13],
        20: [27.13, 48.22, 48.22, 48.22],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax_idx, k in enumerate(ks):
        ax = axes[ax_idx]
        x = np.arange(len(stages))
        width = 0.35

        bars_actual = ax.bar(x - width/2, actual[k], width,
                            label='Actual', color=COLORS['blue'], edgecolor='white')
        bars_oracle = ax.bar(x + width/2, oracle[k], width,
                            label='Oracle', color=COLORS['orange'], alpha=0.7,
                            edgecolor='white')

        # Add gap annotations
        for i in range(len(stages)):
            gap = oracle[k][i] - actual[k][i]
            if gap > 1.0:
                mid_y = (actual[k][i] + oracle[k][i]) / 2
                ax.annotate(f'+{gap:.1f}',
                           xy=(x[i] + width/2, oracle[k][i]),
                           xytext=(x[i] + width/2 + 0.15, oracle[k][i] + 1.5),
                           fontsize=8, color=COLORS['orange'],
                           ha='left', va='bottom',
                           arrowprops=dict(arrowstyle='-', color=COLORS['orange'],
                                          lw=0.5))

        ax.set_xlabel('')
        ax.set_ylabel(f'Hit@{k} (%)')
        ax.set_title(f'Hit@{k}')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylim(0, 55)

        if ax_idx == 0:
            ax.legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Actual vs Oracle Performance Across Pipeline Stages',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_oracle_gap.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_oracle_gap.png'))
    plt.close()
    print("  Saved fig_oracle_gap.pdf/png")


def fig2_feature_importance(output_dir):
    """Horizontal bar chart of reranker feature coefficients."""
    features = [
        ('File modification\nfrequency (log)', 1.0504),
        ('In base prediction', 0.3399),
        ('Position bucket', -0.3361),
        ('Same directory\nas prediction', 0.2670),
        ('Text relevance\nscore', 0.2436),
        ('Co-change score\nto predictions', 0.1834),
        ('Is test file', 0.1333),
        ('Basename length', -0.1117),
        ('Is __init__.py', 0.0515),
        ('Import connected\nto predictions', -0.0489),
    ]

    names = [f[0] for f in features]
    values = [f[1] for f in features]
    colors = [COLORS['blue'] if v > 0 else COLORS['orange'] for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Logistic Regression Coefficient')
    ax.set_title('Learned Reranker Feature Importance\n(5-fold CV, Leak-Free)',
                fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:+.3f}', va='center', ha=ha, fontsize=8, color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_feature_importance.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_feature_importance.png'))
    plt.close()
    print("  Saved fig_feature_importance.pdf/png")


def fig3_bottleneck_decomposition(output_dir):
    """Bottleneck decomposition: where GT files are lost."""
    categories = [
        ('Found in top-5\n(success)', 1058, COLORS['green']),
        ('In top-10\nnot top-5', 275, COLORS['cyan']),
        ('In pool but\nrank > 10', 848, COLORS['orange']),
        ('Not in pool\n(missed)', 4595, COLORS['purple']),
    ]
    labels = [c[0] for c in categories]
    sizes = [c[1] for c in categories]
    colors = [c[2] for c in categories]
    total = sum(sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        textprops={'fontsize': 9},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax1.set_title(f'GT File Disposition (n={total})', fontweight='bold')

    # Right: Structural connectivity of missed files
    missed_categories = [
        ('No structural\nconnection', 85.2),
        ('Same directory', 12.1),
        ('Co-change\nhistory', 10.8),
        ('Import\nrelation', 4.1),
        ('Call graph', 0.2),
    ]
    m_names = [m[0] for m in missed_categories]
    m_values = [m[1] for m in missed_categories]

    bars = ax2.barh(range(len(m_names)), m_values,
                    color=[COLORS['purple'], COLORS['cyan'], COLORS['blue'],
                           COLORS['green'], COLORS['gray']],
                    edgecolor='white', height=0.6)
    ax2.set_yticks(range(len(m_names)))
    ax2.set_yticklabels(m_names, fontsize=9)
    ax2.set_xlabel('% of Missed GT Files')
    ax2.set_title('Structural Connectivity of\n4,595 Missed GT Files', fontweight='bold')
    ax2.invert_yaxis()

    for bar, val in zip(bars, m_values):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_bottleneck.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_bottleneck.png'))
    plt.close()
    print("  Saved fig_bottleneck.pdf/png")


def fig4_signal_ablation(output_dir):
    """Signal ablation: contribution of each expansion signal."""
    signals = ['None\n(SFT only)', 'Co-change', 'Import', 'Directory',
               'Test-src', 'All signals', 'All +\nreranking']
    h5 =  [25.38, 31.10, 30.23, 29.74, 30.62, 31.16, 33.24]
    h10 = [26.54, 36.69, 34.76, 34.42, 34.93, 36.11, 38.57]
    h20 = [27.13, 41.65, 39.65, 40.00, 40.44, 41.44, 42.79]

    x = np.arange(len(signals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, h5, width, label='Hit@5', color=COLORS['blue'], edgecolor='white')
    ax.bar(x, h10, width, label='Hit@10', color=COLORS['orange'], edgecolor='white')
    ax.bar(x + width, h20, width, label='Hit@20', color=COLORS['green'], edgecolor='white')

    # GAT baselines as horizontal lines
    ax.axhline(y=31.51, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(signals) - 0.5, 31.51 + 0.5, 'GAT H@5', fontsize=8, color=COLORS['gray'])
    ax.axhline(y=37.40, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.7)
    ax.text(len(signals) - 0.5, 37.40 + 0.5, 'GAT H@10', fontsize=8, color=COLORS['gray'])

    ax.set_xlabel('')
    ax.set_ylabel('Hit@K (%)')
    ax.set_title('Expansion Signal Ablation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(signals, fontsize=9)
    ax.set_ylim(20, 48)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_signal_ablation.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_signal_ablation.png'))
    plt.close()
    print("  Saved fig_signal_ablation.pdf/png")


def fig5_method_comparison(output_dir):
    """Main comparison bar chart across all methods."""
    methods = [
        'GAT\n(GREPO)',
        'Agentless\n(GPT-4o)',
        'Zero-shot\nQwen-7B',
        'SFT-only\n(Ours)',
        'SFT +\nExpand',
        'SFT +\nRerank',
        'SFT +\nLearned\nRerank',
    ]
    h1 =  [14.80, 13.65, 4.49,  18.73, 18.73, 18.73, 18.73]
    h5 =  [31.51, 21.86, 6.60,  25.38, 31.13, 33.24, 34.66]
    h10 = [37.40, 23.43, 6.62,  26.54, 36.34, 38.57, 40.65]
    h20 = [41.25, 23.43, 6.62,  27.13, 41.27, 42.79, 45.81]

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5*width, h1, width, label='Hit@1', color=COLORS['cyan'], edgecolor='white')
    ax.bar(x - 0.5*width, h5, width, label='Hit@5', color=COLORS['blue'], edgecolor='white')
    ax.bar(x + 0.5*width, h10, width, label='Hit@10', color=COLORS['orange'], edgecolor='white')
    ax.bar(x + 1.5*width, h20, width, label='Hit@20', color=COLORS['green'], edgecolor='white')

    ax.set_ylabel('Hit@K (%)')
    ax.set_title('Method Comparison on GREPO Benchmark', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 52)
    ax.legend(loc='upper right', ncol=4, framealpha=0.9)

    # Highlight our best
    ax.axvspan(3.5, 6.5, alpha=0.05, color=COLORS['blue'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_method_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_method_comparison.png'))
    plt.close()
    print("  Saved fig_method_comparison.pdf/png")


def fig6_negative_transfer(output_dir):
    """Negative transfer: GSP variants vs SFT-only."""
    methods = ['SFT-only\n(no GSP)', 'CoChange\nGSP+SFT', 'AST\nGSP+SFT', 'Combined\nGSP+SFT']
    h1 = [18.73, 13.09, 13.81, 11.94]
    h5 = [25.38, 15.53, 16.54, 15.10]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, h1, width, label='Hit@1', color=COLORS['blue'], edgecolor='white')
    bars2 = ax.bar(x + width/2, h5, width, label='Hit@5', color=COLORS['orange'], edgecolor='white')

    # Mark negative transfer with arrows
    for i in range(1, len(methods)):
        delta_h1 = h1[i] - h1[0]
        ax.annotate(f'{delta_h1:+.1f}', xy=(x[i] - width/2, h1[i]),
                   xytext=(x[i] - width/2, h1[i] + 1.5),
                   ha='center', fontsize=8, color='red', fontweight='bold')
        delta_h5 = h5[i] - h5[0]
        ax.annotate(f'{delta_h5:+.1f}', xy=(x[i] + width/2, h5[i]),
                   xytext=(x[i] + width/2, h5[i] + 1.5),
                   ha='center', fontsize=8, color='red', fontweight='bold')

    ax.set_ylabel('Hit@K (%)')
    ax.set_title('Graph Structure Pre-training Causes Negative Transfer',
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_negative_transfer.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_negative_transfer.png'))
    plt.close()
    print("  Saved fig_negative_transfer.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='docs/figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating publication figures...")
    fig1_oracle_gap(args.output_dir)
    fig2_feature_importance(args.output_dir)
    fig3_bottleneck_decomposition(args.output_dir)
    fig4_signal_ablation(args.output_dir)
    fig5_method_comparison(args.output_dir)
    fig6_negative_transfer(args.output_dir)
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
