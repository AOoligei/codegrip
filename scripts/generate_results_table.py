#!/usr/bin/env python3
"""
Generate comprehensive results tables in both Markdown and LaTeX format.
Reads summary.json files from all experiments and produces publication-ready tables.

Usage:
    python scripts/generate_results_table.py [--output_dir docs/tables]
"""

import json
import os
import sys
import argparse
from collections import OrderedDict

EXPERIMENTS_DIR = "/home/chenlibin/grepo_agent/experiments"
KS = [1, 5, 10, 20]

# Define experiment configurations
EXPERIMENTS = OrderedDict([
    # Baselines
    ("GAT (GREPO)", {"source": "manual", "values": {1: 14.80, 5: 31.51, 10: 37.40, 20: 41.25}}),
    ("GATv2 (GREPO)", {"source": "manual", "values": {1: 14.80, 5: 31.51, 10: 37.40, 20: 41.25}}),
    ("Agentless (GPT-4o)", {"source": "manual", "values": {1: 13.65, 5: 21.86, 10: 23.43, 20: 23.43}}),
    ("Zero-shot Qwen2.5-7B", {"source": "manual", "values": {1: 4.49, 5: 6.60, 10: 6.62, 20: 6.62}}),

    # Our methods
    ("Exp1: SFT-only", {"dir": "exp1_sft_only", "stage": "eval_filetree"}),
    ("Exp1: + expansion", {"dir": "exp1_sft_only", "stage": "eval_unified_expansion"}),
    ("Exp1: + reranking", {"dir": "exp1_sft_only", "stage": "eval_reranked"}),

    ("Exp2: CoChange-GSP+SFT", {"dir": "exp2_cochange_gsp_sft", "stage": "eval_filetree"}),
    ("Exp3: AST-GSP+SFT", {"dir": "exp3_ast_gsp_sft", "stage": "eval_filetree"}),
    ("Exp4: Combined-GSP+SFT", {"dir": "exp4_combined_gsp_sft", "stage": "eval_filetree"}),

    ("Exp5: Coder-7B SFT", {"dir": "exp5_coder_sft_only", "stage": "eval_filetree"}),
    ("Exp5: + expansion", {"dir": "exp5_coder_sft_only", "stage": "eval_unified_expansion"}),
    ("Exp5: + reranking", {"dir": "exp5_coder_sft_only", "stage": "eval_reranked"}),

    ("Exp6: Warm-start SFT", {"dir": "exp6_warmstart_cochange", "stage": "eval_filetree"}),
    ("Exp6: + expansion", {"dir": "exp6_warmstart_cochange", "stage": "eval_unified_expansion"}),
    ("Exp6: + reranking", {"dir": "exp6_warmstart_cochange", "stage": "eval_reranked"}),

    ("Exp7: Multi-task SFT", {"dir": "exp7_multitask_sft", "stage": "eval_filetree"}),
    ("Exp7: + expansion", {"dir": "exp7_multitask_sft", "stage": "eval_unified_expansion"}),
    ("Exp7: + reranking", {"dir": "exp7_multitask_sft", "stage": "eval_reranked"}),

    ("Exp8: Graph-cond SFT", {"dir": "exp8_graph_sft", "stage": "eval_graph"}),
    ("Exp8: + expansion", {"dir": "exp8_graph_sft", "stage": "eval_graph_expansion"}),
    ("Exp8: + reranking", {"dir": "exp8_graph_sft", "stage": "eval_graph_reranked"}),

    ("Exp9: TGS filetree", {"dir": "exp9_tgs_filetree", "stage": "eval_filetree"}),
    ("Exp9: + expansion", {"dir": "exp9_tgs_filetree", "stage": "eval_unified_expansion"}),
    ("Exp9: + reranking", {"dir": "exp9_tgs_filetree", "stage": "eval_reranked"}),

    ("Exp10: TGS+graph", {"dir": "exp10_tgs_graph", "stage": "eval_graph"}),
    ("Exp10: + expansion", {"dir": "exp10_tgs_graph", "stage": "eval_graph_expansion"}),
    ("Exp10: + reranking", {"dir": "exp10_tgs_graph", "stage": "eval_graph_reranked"}),
])


def load_summary(exp_dir: str, stage: str) -> dict:
    """Load summary.json for an experiment stage."""
    path = os.path.join(EXPERIMENTS_DIR, exp_dir, stage, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_results() -> list:
    """Collect all available results."""
    results = []
    for name, config in EXPERIMENTS.items():
        if config.get("source") == "manual":
            values = config["values"]
            results.append({
                "name": name,
                "values": {f"hit@{k}": values[k] for k in KS},
                "available": True,
            })
        else:
            summary = load_summary(config["dir"], config["stage"])
            if summary and "overall" in summary:
                results.append({
                    "name": name,
                    "values": summary["overall"],
                    "available": True,
                })
            else:
                results.append({
                    "name": name,
                    "values": {},
                    "available": False,
                })
    return results


def find_best_values(results: list) -> dict:
    """Find the best value for each metric (excluding baselines)."""
    best = {}
    for k in KS:
        key = f"hit@{k}"
        best_val = -1
        for r in results:
            if not r["available"]:
                continue
            # Skip baselines for "best" marking
            if any(base in r["name"] for base in ["GAT", "Agentless", "Zero-shot"]):
                continue
            val = r["values"].get(key, -1)
            if val > best_val:
                best_val = val
        best[key] = best_val
    return best


def generate_markdown(results: list, best: dict) -> str:
    """Generate Markdown table."""
    lines = []
    lines.append("# Comprehensive Results on GREPO Benchmark\n")
    lines.append(f"| {'Method':<30s} | {'Hit@1':>7s} | {'Hit@5':>7s} | {'Hit@10':>7s} | {'Hit@20':>7s} |")
    lines.append(f"|{'-' * 31}|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 9}|")

    for r in results:
        name = r["name"]
        if not r["available"]:
            lines.append(f"| {name:<30s} |    TBD  |    TBD  |    TBD  |    TBD  |")
            continue

        cells = []
        for k in KS:
            key = f"hit@{k}"
            val = r["values"].get(key, 0)
            is_best = abs(val - best.get(key, -1)) < 0.01 and not any(
                base in name for base in ["GAT", "Agentless", "Zero-shot"]
            )
            if is_best:
                cells.append(f"**{val:5.2f}**")
            else:
                cells.append(f"  {val:5.2f} ")
        lines.append(f"| {name:<30s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    return "\n".join(lines)


def generate_latex(results: list, best: dict) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive results on the GREPO benchmark. "
                 r"Best results (excluding baselines) are \textbf{bolded}.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Hit@1 & Hit@5 & Hit@10 & Hit@20 \\")
    lines.append(r"\midrule")

    prev_was_baseline = False
    for r in results:
        name = r["name"]
        is_baseline = any(base in name for base in ["GAT", "Agentless", "Zero-shot"])

        # Add separator between baselines and our methods
        if prev_was_baseline and not is_baseline:
            lines.append(r"\midrule")
        prev_was_baseline = is_baseline

        if not r["available"]:
            lines.append(f"{name} & TBD & TBD & TBD & TBD \\\\")
            continue

        cells = []
        for k in KS:
            key = f"hit@{k}"
            val = r["values"].get(key, 0)
            is_best = abs(val - best.get(key, -1)) < 0.01 and not is_baseline
            if is_best:
                cells.append(f"\\textbf{{{val:.2f}}}")
            else:
                cells.append(f"{val:.2f}")

        # Escape underscores and special chars in name for LaTeX
        latex_name = name.replace("_", r"\_").replace("#", r"\#")

        # Indent sub-results
        if name.startswith("Exp") and ("expansion" in name or "reranking" in name):
            latex_name = r"\quad " + latex_name.split(": ")[1]

        lines.append(f"{latex_name} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_compact(results: list, best: dict) -> str:
    """Generate compact LaTeX table (main paper version, only key results)."""
    # Filter to key results only
    key_names = [
        "GAT (GREPO)", "Agentless (GPT-4o)", "Zero-shot Qwen2.5-7B",
        "Exp1: SFT-only", "Exp1: + expansion", "Exp1: + reranking",
    ]
    # Add best expansion+reranking from other exps
    for prefix in ["Exp5:", "Exp8:", "Exp9:", "Exp10:"]:
        for suffix in [" + reranking"]:
            name = prefix + suffix
            if any(r["name"] == name and r["available"] for r in results):
                key_names.append(name)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results on GREPO benchmark.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Hit@1 & Hit@5 & Hit@10 & Hit@20 \\")
    lines.append(r"\midrule")

    for r in results:
        if r["name"] not in key_names:
            continue
        if not r["available"]:
            continue

        cells = []
        for k in KS:
            key = f"hit@{k}"
            val = r["values"].get(key, 0)
            is_best = abs(val - best.get(key, -1)) < 0.01 and not any(
                base in r["name"] for base in ["GAT", "Agentless", "Zero-shot"]
            )
            if is_best:
                cells.append(f"\\textbf{{{val:.2f}}}")
            else:
                cells.append(f"{val:.2f}")

        latex_name = r["name"].replace("_", r"\_")
        if any(base in r["name"] for base in ["GAT", "Agentless", "Zero-shot"]):
            pass  # Keep baseline names as-is
        elif "reranking" in r["name"]:
            # Show as "CodeGRIP (Exp X)"
            exp_num = r["name"].split(":")[0]
            latex_name = f"CodeGRIP ({exp_num})"
        elif "expansion" in r["name"]:
            continue  # Skip expansion-only in compact table

        lines.append(f"{latex_name} & {' & '.join(cells)} \\\\")

        # Add midrule after baselines
        if r["name"] == "Zero-shot Qwen2.5-7B":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="docs/tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = collect_results()
    best = find_best_values(results)

    # Count available results
    available = sum(1 for r in results if r["available"])
    total = len(results)
    print(f"Results available: {available}/{total}")

    # Generate tables
    md = generate_markdown(results, best)
    with open(os.path.join(args.output_dir, "results_table.md"), "w") as f:
        f.write(md)
    print(f"\nMarkdown table saved to {args.output_dir}/results_table.md")

    latex_full = generate_latex(results, best)
    with open(os.path.join(args.output_dir, "results_table_full.tex"), "w") as f:
        f.write(latex_full)
    print(f"LaTeX (full) saved to {args.output_dir}/results_table_full.tex")

    latex_compact = generate_latex_compact(results, best)
    with open(os.path.join(args.output_dir, "results_table_compact.tex"), "w") as f:
        f.write(latex_compact)
    print(f"LaTeX (compact) saved to {args.output_dir}/results_table_compact.tex")

    # Print current markdown table to stdout
    print(f"\n{md}")


if __name__ == "__main__":
    main()
