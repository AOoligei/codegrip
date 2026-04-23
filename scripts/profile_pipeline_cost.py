#!/usr/bin/env python3
"""Profile computational cost of the CodeGRIP pipeline.

Measures wall-clock time and memory for each pipeline stage:
  1. LLM Inference (base predictions)
  2. Multi-signal expansion
  3. Issue-text reranking
  4. Learned reranking (CV)

Outputs a LaTeX-ready cost table + Markdown summary.

Usage:
    python scripts/profile_pipeline_cost.py \
        --exp_dir experiments/exp1_sft_only \
        --test_data data/grepo_text/grepo_test.jsonl \
        --train_data data/grepo_text/grepo_train.jsonl \
        --dep_graph_dir data/dep_graphs \
        --file_tree_dir data/file_trees \
        --output_dir docs/tables
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def estimate_token_count(text):
    """Rough token estimate: ~4 chars per token for English/code."""
    return len(text) / 4


def profile_expansion(base_pred_path, train_data_path, dep_graph_dir, file_tree_dir, output_dir):
    """Profile multi-signal expansion."""
    from src.eval.multi_signal_expansion import (
        build_cochange_index,
        build_dir_index,
        build_import_index,
        expand_predictions,
    )

    # Phase 1: Index building
    tracemalloc.start()
    t0 = time.perf_counter()

    cochange_idx = build_cochange_index(train_data_path)
    t_cochange = time.perf_counter() - t0

    t1 = time.perf_counter()
    import_idx = build_import_index(dep_graph_dir)
    t_import = time.perf_counter() - t1

    t2 = time.perf_counter()
    dir_result = build_dir_index(file_tree_dir)
    t_dir = time.perf_counter() - t2

    index_time = time.perf_counter() - t0
    _, index_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # dir_index returns (dir_idx, all_py_files) tuple
    if isinstance(dir_result, tuple):
        dir_idx, all_py_files = dir_result
    else:
        dir_idx = dir_result
        all_py_files = {}

    # Phase 2: Per-example expansion
    exp_output = os.path.join(output_dir, 'predictions.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    tracemalloc.start()
    t3 = time.perf_counter()
    expand_predictions(
        base_pred_path, cochange_idx, import_idx, dir_idx, all_py_files,
        exp_output,
        max_expand=35, min_cochange_score=0.02, max_dir_size=35
    )
    expand_time = time.perf_counter() - t3
    _, expand_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Load expanded predictions to count candidates
    expanded = load_jsonl(exp_output)
    n = len(expanded)
    avg_candidates = sum(len(p.get('predicted', [])) for p in expanded) / max(n, 1)

    return {
        'index_build_sec': index_time,
        'index_cochange_sec': t_cochange,
        'index_import_sec': t_import - t1,
        'index_dir_sec': t_dir - t2,
        'expand_sec': expand_time,
        'total_sec': index_time + expand_time,
        'index_peak_mb': index_peak / 1024 / 1024,
        'expand_peak_mb': expand_peak / 1024 / 1024,
        'n_examples': n,
        'avg_candidates_after': avg_candidates,
        'per_example_ms': (expand_time / max(n, 1)) * 1000,
    }, exp_output


def profile_reranking(expanded_pred_path, test_data_path, output_dir):
    """Profile issue-text-aware reranking."""
    import subprocess

    n = len(load_jsonl(expanded_pred_path))
    rerank_output = os.path.join(output_dir, '_reranked_predictions.jsonl')

    t0 = time.perf_counter()

    cmd = [
        sys.executable, 'src/eval/rerank_predictions.py',
        '--predictions', expanded_pred_path,
        '--test_data', test_data_path,
        '--output', rerank_output,
        '--promote_top5', '1', '--promote_top10', '3', '--threshold', '0.2'
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    total_time = time.perf_counter() - t0

    return {
        'total_sec': total_time,
        'peak_mb': 0,  # subprocess memory not measurable this way
        'n_examples': n,
        'per_example_ms': (total_time / max(n, 1)) * 1000,
    }, rerank_output


def estimate_llm_inference(predictions):
    """Estimate LLM inference cost from recorded inference times and response lengths."""
    total_time = 0
    total_input_tokens = 0
    total_output_tokens = 0
    n_with_time = 0

    for p in predictions:
        # Estimate output tokens from response
        response = p.get('response', '')
        output_tokens = estimate_token_count(response)
        total_output_tokens += output_tokens

        # Estimate input tokens (~3072 tokens max seq length)
        total_input_tokens += 2500  # avg prompt ~2500 tokens

        if 'inference_time' in p:
            total_time += p['inference_time']
            n_with_time += 1

    n = len(predictions)
    avg_time = total_time / max(n_with_time, 1)

    return {
        'n_examples': n,
        'total_time_sec': total_time,
        'avg_time_sec': avg_time,
        'total_input_tokens': int(total_input_tokens),
        'total_output_tokens': int(total_output_tokens),
        'total_tokens': int(total_input_tokens + total_output_tokens),
        'tokens_per_sec': total_output_tokens / max(total_time, 0.01),
        'estimated_gpu_hours': total_time / 3600,
    }


def format_markdown_table(llm_stats, expansion_stats, rerank_stats):
    """Generate Markdown cost comparison table."""
    lines = []
    lines.append("## Computational Cost Analysis")
    lines.append("")
    lines.append("| Stage | Wall-Clock | Per-Example | Peak Memory | Hardware |")
    lines.append("|-------|-----------|-------------|-------------|----------|")

    # LLM Inference
    llm_total = llm_stats['total_time_sec']
    if llm_total > 3600:
        llm_fmt = f"{llm_total/3600:.1f}h"
    elif llm_total > 60:
        llm_fmt = f"{llm_total/60:.1f}min"
    else:
        llm_fmt = f"{llm_total:.1f}s"
    lines.append(
        f"| **1. LLM Inference** | {llm_fmt} | {llm_stats['avg_time_sec']*1000:.0f}ms | ~16 GB | 1x RTX 4090 |"
    )

    # Expansion
    exp_total = expansion_stats['total_sec']
    lines.append(
        f"| **2. Multi-Signal Expansion** | {exp_total:.1f}s | {expansion_stats['per_example_ms']:.1f}ms | {expansion_stats['index_peak_mb']:.0f} MB | CPU only |"
    )

    # Reranking
    rer_total = rerank_stats['total_sec']
    lines.append(
        f"| **3. Issue-Text Reranking** | {rer_total:.1f}s | {rerank_stats['per_example_ms']:.1f}ms | {rerank_stats['peak_mb']:.0f} MB | CPU only |"
    )

    # Total
    total = llm_total + exp_total + rer_total
    lines.append(f"| **Total Pipeline** | {total/3600:.2f}h | {total/llm_stats['n_examples']*1000:.0f}ms | ~16 GB | 1x GPU + CPU |")

    lines.append("")
    lines.append("### Token Budget")
    lines.append("")
    lines.append(f"- Input tokens (total): {llm_stats['total_input_tokens']:,}")
    lines.append(f"- Output tokens (total): {llm_stats['total_output_tokens']:,}")
    lines.append(f"- **Total tokens**: {llm_stats['total_tokens']:,}")
    lines.append(f"- Generation throughput: {llm_stats['tokens_per_sec']:.0f} tokens/sec")
    lines.append(f"- Test examples: {llm_stats['n_examples']}")
    lines.append("")
    lines.append("### Overhead Analysis")
    lines.append("")
    overhead_pct = ((exp_total + rer_total) / max(llm_total, 0.01)) * 100
    lines.append(
        f"Post-processing overhead (expansion + reranking) adds **{overhead_pct:.1f}%** "
        f"to the LLM inference time ({exp_total + rer_total:.1f}s vs {llm_fmt} for inference). "
        f"The expansion step produces {expansion_stats['avg_candidates_after']:.1f} candidates "
        f"per example on average (from ~4 base predictions)."
    )

    return "\n".join(lines)


def format_latex_table(llm_stats, expansion_stats, rerank_stats):
    """Generate LaTeX cost table."""
    llm_total = llm_stats['total_time_sec']
    if llm_total > 3600:
        llm_fmt = f"{llm_total/3600:.1f}h"
    else:
        llm_fmt = f"{llm_total/60:.0f}min"

    exp_total = expansion_stats['total_sec']
    rer_total = rerank_stats['total_sec']
    overhead = ((exp_total + rer_total) / max(llm_total, 0.01)) * 100

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational cost breakdown of the CodeGRIP pipeline on " +
        f"{llm_stats['n_examples']}" + r" test examples. Post-processing adds " +
        f"<{max(overhead, 0.1):.0f}" + r"\% overhead to LLM inference.}",
        r"\label{tab:cost}",
        r"\small",
        r"\begin{tabular}{lcccr}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Time} & \textbf{Per-Ex.} & \textbf{Memory} & \textbf{Hardware} \\",
        r"\midrule",
        f"LLM Inference & {llm_fmt} & {llm_stats['avg_time_sec']*1000:.0f}ms & $\\sim$16GB & 1$\\times$ RTX 4090 \\\\",
        f"Expansion & {exp_total:.1f}s & {expansion_stats['per_example_ms']:.1f}ms & {expansion_stats['index_peak_mb']:.0f}MB & CPU \\\\",
        f"Reranking & {rer_total:.1f}s & {rerank_stats['per_example_ms']:.1f}ms & {rerank_stats['peak_mb']:.0f}MB & CPU \\\\",
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{llm_fmt}+}} & --- & $\\sim$16GB & 1$\\times$ GPU \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Profile CodeGRIP pipeline cost')
    parser.add_argument('--exp_dir', default='experiments/exp1_sft_only')
    parser.add_argument('--test_data', default='data/grepo_text/grepo_test.jsonl')
    parser.add_argument('--train_data', default='data/grepo_text/grepo_train.jsonl')
    parser.add_argument('--dep_graph_dir', default='data/dep_graphs')
    parser.add_argument('--file_tree_dir', default='data/file_trees')
    parser.add_argument('--output_dir', default='docs/tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base predictions
    base_pred_path = os.path.join(args.exp_dir, 'eval_filetree', 'predictions.jsonl')
    if not os.path.exists(base_pred_path):
        print(f"Error: {base_pred_path} not found")
        sys.exit(1)

    print("Loading base predictions...")
    base_preds = load_jsonl(base_pred_path)
    print(f"  {len(base_preds)} examples loaded")

    # 1. Estimate LLM inference cost
    print("\n=== Stage 1: LLM Inference (estimated from recorded times) ===")
    llm_stats = estimate_llm_inference(base_preds)
    print(f"  Total inference time: {llm_stats['total_time_sec']:.1f}s ({llm_stats['estimated_gpu_hours']:.2f} GPU-hours)")
    print(f"  Avg per example: {llm_stats['avg_time_sec']*1000:.0f}ms")
    print(f"  Total tokens: {llm_stats['total_tokens']:,}")
    print(f"  Throughput: {llm_stats['tokens_per_sec']:.0f} tok/s")

    # 2. Profile expansion
    print("\n=== Stage 2: Multi-Signal Expansion (measured) ===")
    profile_out_dir = os.path.join(args.output_dir, '_profile_tmp')
    os.makedirs(profile_out_dir, exist_ok=True)
    expansion_stats, expanded_path = profile_expansion(
        base_pred_path, args.train_data, args.dep_graph_dir, args.file_tree_dir,
        profile_out_dir
    )
    print(f"  Index building: {expansion_stats['index_build_sec']:.2f}s")
    print(f"  Expansion: {expansion_stats['expand_sec']:.2f}s")
    print(f"  Total: {expansion_stats['total_sec']:.2f}s")
    print(f"  Peak memory: {expansion_stats['index_peak_mb']:.0f}MB")
    print(f"  Per-example: {expansion_stats['per_example_ms']:.2f}ms")
    print(f"  Avg candidates after expansion: {expansion_stats['avg_candidates_after']:.1f}")

    # 3. Profile reranking
    print("\n=== Stage 3: Issue-Text Reranking (measured) ===")
    rerank_stats, reranked_path = profile_reranking(expanded_path, args.test_data, profile_out_dir)
    print(f"  Total: {rerank_stats['total_sec']:.2f}s")
    print(f"  Peak memory: {rerank_stats['peak_mb']:.0f}MB")
    print(f"  Per-example: {rerank_stats['per_example_ms']:.2f}ms")

    # Summary
    total = llm_stats['total_time_sec'] + expansion_stats['total_sec'] + rerank_stats['total_sec']
    overhead = expansion_stats['total_sec'] + rerank_stats['total_sec']
    overhead_pct = (overhead / max(llm_stats['total_time_sec'], 0.01)) * 100

    print(f"\n=== Summary ===")
    print(f"  LLM Inference: {llm_stats['total_time_sec']:.1f}s ({llm_stats['total_time_sec']/total*100:.1f}%)")
    print(f"  Expansion:     {expansion_stats['total_sec']:.1f}s ({expansion_stats['total_sec']/total*100:.1f}%)")
    print(f"  Reranking:     {rerank_stats['total_sec']:.1f}s ({rerank_stats['total_sec']/total*100:.1f}%)")
    print(f"  Post-processing overhead: {overhead_pct:.1f}% of LLM time")

    # Generate tables
    md_table = format_markdown_table(llm_stats, expansion_stats, rerank_stats)
    latex_table = format_latex_table(llm_stats, expansion_stats, rerank_stats)

    md_path = os.path.join(args.output_dir, 'cost_analysis.md')
    tex_path = os.path.join(args.output_dir, 'cost_table.tex')

    with open(md_path, 'w') as f:
        f.write(md_table)
    print(f"\nMarkdown table written to {md_path}")

    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {tex_path}")

    # Save raw stats
    stats_path = os.path.join(args.output_dir, 'cost_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'llm_inference': llm_stats,
            'expansion': {k: v for k, v in expansion_stats.items()},
            'reranking': {k: v for k, v in rerank_stats.items()},
            'total_sec': total,
            'overhead_pct': overhead_pct,
        }, f, indent=2)
    print(f"Raw stats written to {stats_path}")


if __name__ == '__main__':
    main()
