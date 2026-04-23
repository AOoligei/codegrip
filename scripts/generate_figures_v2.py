#!/usr/bin/env python3
"""
Generate publication-quality figures for CodeGRIP NeurIPS paper (v2).

Includes:
1. Main results comparison (SFT + RankFT + baselines)
2. RankFT ablation: negative mining strategies
3. Content-aware vs path-only comparison
4. SWE-bench cross-benchmark generalization
5. Bottleneck decomposition (updated)
6. Negative transfer analysis
7. Training loss curves for RankFT runs

Auto-detects which results are available and generates all possible figures.

Usage:
    python scripts/generate_figures_v2.py [--output_dir docs/figures]
"""

import json
import os
import glob
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
    'legend.fontsize': 9,
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
    'yellow': '#CCBB44',
    'grey': '#BBBBBB',
    'red': '#CC3311',
}

ROOT = "/home/chenlibin/grepo_agent/experiments"


def load_summary(path):
    """Load summary.json, handling multiple formats."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    overall = data.get("overall", data.get("metrics", data))
    return {
        "hit@1": overall.get("hit@1", overall.get("hit_at_1", 0)),
        "hit@3": overall.get("hit@3", overall.get("hit_at_3", 0)),
        "hit@5": overall.get("hit@5", overall.get("hit_at_5", 0)),
        "hit@10": overall.get("hit@10", overall.get("hit_at_10", 0)),
        "hit@20": overall.get("hit@20", overall.get("hit_at_20", 0)),
    }


def load_training_loss(log_path):
    """Parse training log for loss curve."""
    steps, losses, avg_losses = [], [], []
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            if "Loss:" in line and "avg:" in line:
                try:
                    parts = line.strip()
                    step = int(parts.split("Step ")[1].split("/")[0])
                    loss = float(parts.split("Loss: ")[1].split(" ")[0])
                    avg = float(parts.split("avg: ")[1].split(")")[0])
                    steps.append(step)
                    losses.append(loss)
                    avg_losses.append(avg)
                except (ValueError, IndexError):
                    continue
    if not steps:
        return None
    return {"steps": steps, "losses": losses, "avg_losses": avg_losses}


def fig_main_results(output_dir):
    """Figure 1: Main comparison across all methods on GREPO."""
    methods = []

    # External baselines
    methods.append(("GATv2\n(GREPO)", {"hit@1": 14.80, "hit@5": 31.51, "hit@10": 37.40, "hit@20": 41.25}, "grey"))
    methods.append(("Agentless\n(GPT-4o)", {"hit@1": 13.65, "hit@5": 21.86, "hit@10": 23.43, "hit@20": 23.43}, "grey"))

    # BM25 baselines
    bm25 = load_summary(os.path.join(ROOT, "baselines/grepo_bm25_improved/summary.json"))
    if bm25:
        methods.append(("BM25\n(improved)", bm25, "grey"))

    # Zero-shot
    zs = load_summary(os.path.join(ROOT, "zeroshot_qwen25_7b_full/summary.json"))
    if zs:
        methods.append(("Zero-shot\nQwen-7B", zs, "cyan"))

    # SFT experiments
    for exp_id, label, color in [
        ("exp1_sft_only", "SFT-only\n(Exp1)", "blue"),
        ("exp5_coder_sft_only", "Coder-SFT\n(Exp5)", "blue"),
        ("exp7_multitask_sft", "Multitask\n(Exp7)", "blue"),
        ("exp8_graph_sft", "Graph-SFT\n(Exp8)", "green"),
        ("exp9_tgs_filetree", "TGS-FT\n(Exp9)", "green"),
        ("exp10_tgs_graph", "TGS-Graph\n(Exp10)", "green"),
        ("exp11_navcot", "NavCoT\n(Exp11)", "green"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{exp_id}/eval_filetree/summary.json"))
        if result:
            methods.append((label, result, color))

    # RankFT results
    for run_id, label, color in [
        ("rankft_runA", "RankFT-A\n(BM25-hard)", "orange"),
        ("rankft_runB", "RankFT-B\n(Graph neg)", "orange"),
        ("rankft_runC", "RankFT-C\n(Random)", "orange"),
        ("rankft_runD", "RankFT-D\n(Content)", "red"),
        ("rankft_runE", "RankFT-E\n(Content-fresh)", "red"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{run_id}_grepo_k200/summary.json"))
        if result:
            methods.append((label, result, color))

    if len(methods) < 3:
        print("  [skip] fig_main_results: not enough data")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 0.9), 5))
    x = np.arange(len(methods))
    width = 0.22
    metrics = ["hit@1", "hit@5", "hit@10"]
    metric_labels = ["Hit@1", "Hit@5", "Hit@10"]

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        values = [m[1].get(metric, 0) for m in methods]
        colors_list = [COLORS[m[2]] for m in methods]
        bars = ax.bar(x + (i - 1) * width, values, width, label=mlabel, alpha=0.85,
                      color=[COLORS[['blue', 'orange', 'green'][i]]] * len(methods),
                      edgecolor='white', linewidth=0.5)

    ax.set_ylabel("Score (%)")
    ax.set_title("File-Level Bug Localization on GREPO Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], fontsize=8, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, max(m[1].get("hit@10", 0) for m in methods) * 1.15)

    path = os.path.join(output_dir, "fig_main_results")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def fig_rankft_ablation(output_dir):
    """Figure 2: RankFT negative mining ablation."""
    runs = []
    for run_id, label in [
        ("rankft_runA", "BM25-hard\n(100%)"),
        ("rankft_runB", "Mixed\n(50/25/25)"),
        ("rankft_runC", "Random\n(100%)"),
        ("rankft_runD", "Content\n(BM25+Graph)"),
        ("rankft_runE", "Content\n(Fresh LoRA)"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{run_id}_grepo_k200/summary.json"))
        if result:
            runs.append((label, result))

    if len(runs) < 2:
        print("  [skip] fig_rankft_ablation: not enough data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(runs))
    width = 0.25

    for i, (metric, mlabel) in enumerate(zip(["hit@1", "hit@5", "hit@10"],
                                              ["Hit@1", "Hit@5", "Hit@10"])):
        values = [r[1].get(metric, 0) for r in runs]
        ax.bar(x + (i - 1) * width, values, width, label=mlabel,
               color=COLORS[['blue', 'orange', 'green'][i]], alpha=0.85, edgecolor='white')

    ax.set_ylabel("Score (%)")
    ax.set_title("RankFT: Effect of Negative Mining Strategy (GREPO)")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in runs])
    ax.legend()

    path = os.path.join(output_dir, "fig_rankft_ablation")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def fig_swebench_comparison(output_dir):
    """Figure 3: SWE-bench cross-benchmark results."""
    methods = []

    # BM25 baseline
    bm25 = load_summary(os.path.join(ROOT, "baselines/swebench_lite_bm25/bm25_path/summary.json"))
    if bm25:
        methods.append(("BM25", bm25, "grey"))

    # SFT
    sft = load_summary(os.path.join(ROOT, "exp1_sft_only/eval_swebench_lite/summary.json"))
    if sft:
        methods.append(("SFT (Exp1)", sft, "blue"))

    # RankFT runs on SWE-bench
    for run_id, label, color in [
        ("rankft_runA", "RankFT-A", "orange"),
        ("rankft_runB", "RankFT-B", "orange"),
        ("rankft_runD", "RankFT-D (Content)", "red"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{run_id}_swebench_k200/summary.json"))
        if result:
            methods.append((label, result, color))

    if len(methods) < 2:
        print("  [skip] fig_swebench_comparison: not enough data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.3

    for i, (metric, mlabel) in enumerate(zip(["hit@1", "hit@5"], ["Hit@1", "Hit@5"])):
        values = [m[1].get(metric, 0) for m in methods]
        ax.bar(x + (i - 0.5) * width, values, width, label=mlabel,
               color=COLORS[['blue', 'orange'][i]], alpha=0.85, edgecolor='white')

    ax.set_ylabel("Score (%)")
    ax.set_title("Cross-Benchmark Generalization (SWE-bench Lite)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods])
    ax.legend()

    path = os.path.join(output_dir, "fig_swebench")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def fig_training_curves(output_dir):
    """Figure 4: RankFT training loss curves."""
    runs = []
    for run_id, label, color in [
        ("rankft_runA_bm25only", "Run A (BM25-hard)", COLORS['blue']),
        ("rankft_runB_graph", "Run B (Graph neg)", COLORS['orange']),
        ("rankft_runC_random", "Run C (Random)", COLORS['green']),
        ("rankft_runD_content", "Run D (Content)", COLORS['red']),
        ("rankft_runE_content_fresh", "Run E (Content fresh)", COLORS['purple']),
    ]:
        data = load_training_loss(os.path.join(ROOT, f"{run_id}/train.log"))
        if data:
            runs.append((label, data, color))

    if not runs:
        print("  [skip] fig_training_curves: no training logs found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color in runs:
        ax.plot(data["steps"], data["avg_losses"], label=label, color=color, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Average Loss")
    ax.set_title("RankFT Training Loss Curves")
    ax.legend()

    path = os.path.join(output_dir, "fig_training_curves")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def fig_negative_transfer(output_dir):
    """Figure 5: Negative transfer analysis."""
    methods = [
        ("SFT-only\n(baseline)", {"hit@1": 18.73, "hit@5": 25.38, "hit@10": 26.54}),
        ("Co-change\nGSP", {"hit@1": 13.09, "hit@5": 15.53, "hit@10": 15.69}),
        ("AST\nGSP", {"hit@1": 13.81, "hit@5": 16.54, "hit@10": 17.01}),
        ("Combined\nGSP", {"hit@1": 11.94, "hit@5": 15.10, "hit@10": 15.41}),
    ]

    # Add exp6/7 if available
    for exp_id, label in [
        ("exp6_warmstart_cochange", "Warmstart\nco-change"),
        ("exp7_multitask_sft", "Multitask\nSFT"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{exp_id}/eval_filetree/summary.json"))
        if result:
            methods.append((label, result))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.25

    for i, (metric, mlabel, color) in enumerate(zip(
        ["hit@1", "hit@5", "hit@10"],
        ["Hit@1", "Hit@5", "Hit@10"],
        [COLORS['blue'], COLORS['orange'], COLORS['green']],
    )):
        values = [m[1].get(metric, 0) for m in methods]
        ax.bar(x + (i - 1) * width, values, width, label=mlabel, color=color, alpha=0.85, edgecolor='white')

    # Baseline line
    ax.axhline(y=18.73, color='black', linestyle='--', alpha=0.4, linewidth=0.8, label='SFT baseline (Hit@1)')

    ax.set_ylabel("Score (%)")
    ax.set_title("Negative Transfer: Graph Pre-training Variants")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax.legend(loc='upper right')

    path = os.path.join(output_dir, "fig_negative_transfer")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def fig_graph_sft_comparison(output_dir):
    """Figure 6: Graph-conditioned SFT experiments comparison."""
    methods = []

    for exp_id, label, color in [
        ("exp1_sft_only", "SFT-only", "grey"),
        ("exp5_coder_sft_only", "Coder-SFT", "blue"),
        ("exp7_multitask_sft", "Multitask", "blue"),
        ("exp8_graph_sft", "Graph-cond.", "green"),
        ("exp9_tgs_filetree", "TGS-FT", "green"),
        ("exp10_tgs_graph", "TGS-Graph", "green"),
        ("exp11_navcot", "NavCoT", "purple"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{exp_id}/eval_filetree/summary.json"))
        if result:
            methods.append((label, result, color))

    if len(methods) < 3:
        print("  [skip] fig_graph_sft_comparison: not enough data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.25

    for i, (metric, mlabel) in enumerate(zip(["hit@1", "hit@5", "hit@10"],
                                              ["Hit@1", "Hit@5", "Hit@10"])):
        values = [m[1].get(metric, 0) for m in methods]
        ax.bar(x + (i - 1) * width, values, width, label=mlabel,
               color=COLORS[['blue', 'orange', 'green'][i]], alpha=0.85, edgecolor='white')

    ax.axhline(y=18.73, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_ylabel("Score (%)")
    ax.set_title("SFT Variants: Graph Knowledge Integration Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods])
    ax.legend()

    path = os.path.join(output_dir, "fig_graph_sft")
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved {path}.pdf/png")


def generate_latex_table(output_dir):
    """Generate LaTeX results table."""
    rows = []

    # Baselines
    rows.append(("\\midrule", None))
    rows.append(("\\multicolumn{5}{l}{\\textit{Baselines}} \\\\", None))
    rows.append(("GATv2 (GREPO)", {"hit@1": 14.80, "hit@5": 31.51, "hit@10": 37.40, "hit@20": 41.25}))
    rows.append(("Agentless (GPT-4o)", {"hit@1": 13.65, "hit@5": 21.86, "hit@10": 23.43, "hit@20": 23.43}))

    bm25 = load_summary(os.path.join(ROOT, "baselines/grepo_bm25_improved/summary.json"))
    if bm25:
        rows.append(("BM25 (improved)", bm25))

    zs = load_summary(os.path.join(ROOT, "zeroshot_qwen25_7b_full/summary.json"))
    if zs:
        rows.append(("Zero-shot Qwen2.5-7B", zs))

    # SFT methods
    rows.append(("\\midrule", None))
    rows.append(("\\multicolumn{5}{l}{\\textit{SFT Methods (Ours)}} \\\\", None))

    for exp_id, label in [
        ("exp1_sft_only", "SFT-only (base)"),
        ("exp5_coder_sft_only", "Coder-SFT"),
        ("exp7_multitask_sft", "Multitask SFT"),
        ("exp8_graph_sft", "Graph-conditioned SFT"),
        ("exp9_tgs_filetree", "TGS-Filetree"),
        ("exp10_tgs_graph", "TGS-Graph"),
        ("exp11_navcot", "NavCoT"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{exp_id}/eval_filetree/summary.json"))
        if result:
            rows.append((label, result))

    # RankFT methods
    rows.append(("\\midrule", None))
    rows.append(("\\multicolumn{5}{l}{\\textit{RankFT Reranking (Ours)}} \\\\", None))

    for run_id, label in [
        ("rankft_runA", "RankFT: BM25-hard neg"),
        ("rankft_runB", "RankFT: Graph neg"),
        ("rankft_runC", "RankFT: Random neg"),
        ("rankft_runD", "RankFT: Content-aware"),
        ("rankft_runE", "RankFT: Content (fresh)"),
    ]:
        result = load_summary(os.path.join(ROOT, f"{run_id}_grepo_k200/summary.json"))
        if result:
            rows.append((label, result))

    # Build LaTeX
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{File-level bug localization on GREPO benchmark.}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Hit@1 & Hit@5 & Hit@10 & Hit@20 \\\\",
    ]

    for item in rows:
        if item[1] is None:
            lines.append(item[0])
        else:
            label, r = item
            lines.append(f"{label} & {r.get('hit@1', 0):.2f} & {r.get('hit@5', 0):.2f} & "
                         f"{r.get('hit@10', 0):.2f} & {r.get('hit@20', 0):.2f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex_path = os.path.join(output_dir, "results_table_v2.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {tex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating figures in {args.output_dir}/...")

    fig_main_results(args.output_dir)
    fig_rankft_ablation(args.output_dir)
    fig_swebench_comparison(args.output_dir)
    fig_training_curves(args.output_dir)
    fig_negative_transfer(args.output_dir)
    fig_graph_sft_comparison(args.output_dir)
    generate_latex_table(args.output_dir)

    print("\nDone! Re-run when new results arrive.")


if __name__ == "__main__":
    main()
