#!/usr/bin/env python3
"""Aggregate all experiment results into a clean comparison table.

Usage:
    python scripts/aggregate_results.py
"""
import json
import os
import glob

ROOT = "/home/chenlibin/grepo_agent/experiments"


def load_summary(path):
    """Load summary.json, handling multiple formats."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Try different key structures
    overall = data.get("overall", data.get("metrics", data))
    return {
        "hit@1": overall.get("hit@1", overall.get("hit_at_1", 0)),
        "hit@5": overall.get("hit@5", overall.get("hit_at_5", 0)),
        "hit@10": overall.get("hit@10", overall.get("hit_at_10", 0)),
        "acc@1": overall.get("acc@1", overall.get("acc_at_1", 0)),
        "n_samples": overall.get("n_samples", overall.get("total", data.get("num_examples", 0))),
    }


def check_status(exp_dir):
    """Check experiment status."""
    if os.path.exists(os.path.join(exp_dir, "stage2_sft/final")):
        return "trained"
    # Check for latest checkpoint
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "stage2_sft/checkpoint-*")))
    if ckpts:
        last = os.path.basename(ckpts[-1]).replace("checkpoint-", "")
        return f"training (ckpt {last})"
    # RankFT experiments
    if os.path.exists(os.path.join(exp_dir, "final")):
        return "trained"
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoint-*")))
    if ckpts:
        last = os.path.basename(ckpts[-1]).replace("checkpoint-", "")
        return f"training (ckpt {last})"
    return "not started"


def main():
    print("=" * 100)
    print("CodeGRIP Experiment Results Summary")
    print("=" * 100)

    # ============================================================
    # Section 1: SFT Experiments (GREPO eval)
    # ============================================================
    print("\n## SFT Experiments (GREPO, filetree mode)")
    print(f"{'Experiment':<35} {'Status':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
    print("-" * 85)

    sft_exps = [
        ("exp1_sft_only", "Exp1: SFT only (baseline)"),
        ("exp2_cochange_gsp_sft", "Exp2: Co-change GSP"),
        ("exp3_ast_gsp_sft", "Exp3: AST GSP"),
        ("exp4_combined_gsp_sft", "Exp4: Combined GSP"),
        ("exp5_coder_sft_only", "Exp5: Coder SFT"),
        ("exp6_warmstart_cochange", "Exp6: Warmstart co-change"),
        ("exp7_multitask_sft", "Exp7: Multitask SFT"),
        ("exp8_graph_sft", "Exp8: Graph-conditioned SFT"),
        ("exp9_tgs_filetree", "Exp9: TGS filetree"),
        ("exp10_tgs_graph", "Exp10: TGS graph"),
        ("exp11_navcot", "Exp11: NavCoT"),
    ]

    for exp_id, exp_name in sft_exps:
        exp_dir = os.path.join(ROOT, exp_id)
        status = check_status(exp_dir)

        # Check filetree eval first, then graph eval
        result = load_summary(os.path.join(exp_dir, "eval_filetree/summary.json"))
        mode = "filetree"
        if not result:
            result = load_summary(os.path.join(exp_dir, "eval_graph/summary.json"))
            mode = "graph"
        if result:
            label = f"eval done ({mode})"
            print(f"  {exp_name:<33} {label:<20} {result['hit@1']:>7.2f}% {result['hit@5']:>7.2f}% {result['hit@10']:>7.2f}%")
        else:
            print(f"  {exp_name:<33} {status:<20} {'--':>8} {'--':>8} {'--':>8}")

    # ============================================================
    # Section 2: RankFT Experiments
    # ============================================================
    print("\n## RankFT Experiments (BM25 top-200 reranking)")

    rankft_runs = [
        ("rankft_runA_bm25only", "Run A: BM25-hard neg"),
        ("rankft_runA_best", "Run A: best ckpt"),
        ("rankft_runB_graph", "Run B: Graph neg"),
        ("rankft_runB_best", "Run B: best ckpt"),
        ("rankft_runC_random", "Run C: Random neg"),
        ("rankft_runD_content", "Run D: Content-aware"),
        ("rankft_runD_best", "Run D: best ckpt"),
        ("rankft_runE_content_fresh", "Run E: Content fresh"),
        ("rankft_runF_content_exp6init", "Run F: Content+Exp6 init"),
    ]

    for dataset_label, dataset_suffix in [("GREPO", "grepo_k200"), ("SWE-bench Lite", "swebench_k200")]:
        print(f"\n  ### {dataset_label}")
        print(f"  {'Run':<35} {'Status':<20} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
        print("  " + "-" * 83)

        for run_id, run_name in rankft_runs:
            train_dir = os.path.join(ROOT, run_id)
            eval_dir = os.path.join(ROOT, f"{run_id}_{dataset_suffix}")
            result = load_summary(os.path.join(eval_dir, "summary.json"))
            status = check_status(train_dir)

            if result:
                print(f"    {run_name:<33} {'eval done':<20} {result['hit@1']:>7.2f}% {result['hit@5']:>7.2f}% {result['hit@10']:>7.2f}%")
            else:
                print(f"    {run_name:<33} {status:<20} {'--':>8} {'--':>8} {'--':>8}")

    # ============================================================
    # Section 3: Baselines
    # ============================================================
    print("\n## Baselines")
    print(f"  {'Baseline':<35} {'Dataset':<15} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
    print("  " + "-" * 70)

    baselines = [
        ("baselines/bm25_path", "GREPO", "BM25 (path)"),
        ("baselines/tfidf_path", "GREPO", "TF-IDF (path)"),
        ("baselines/frequency", "GREPO", "Frequency"),
        ("baselines/combined", "GREPO", "Combined retrieval"),
        ("baselines/grepo_bm25_improved", "GREPO", "BM25 improved"),
        ("baselines/swebench_lite_bm25/bm25_path", "SWE-bench", "BM25 (path)"),
        ("baselines/swebench_lite_bm25/combined", "SWE-bench", "Combined"),
        ("zeroshot_qwen25_7b_full", "GREPO", "Zero-shot Qwen2.5-7B"),
    ]

    for exp_path, dataset, name in baselines:
        result = load_summary(os.path.join(ROOT, exp_path, "summary.json"))
        if result:
            print(f"    {name:<33} {dataset:<15} {result['hit@1']:>7.2f}% {result['hit@5']:>7.2f}% {result['hit@10']:>7.2f}%")

    # ============================================================
    # Section 4: SWE-bench SFT results
    # ============================================================
    print("\n## SWE-bench SFT Results")
    swebench_evals = [
        ("exp1_sft_only/eval_swebench_lite", "Exp1 SFT"),
        ("exp1_sft_only/eval_swebench_ensemble_content", "Exp1 ensemble+content"),
    ]
    print(f"  {'Experiment':<35} {'Hit@1':>8} {'Hit@5':>8}")
    print("  " + "-" * 55)
    for exp_path, name in swebench_evals:
        result = load_summary(os.path.join(ROOT, exp_path, "summary.json"))
        if result:
            print(f"    {name:<33} {result['hit@1']:>7.2f}% {result['hit@5']:>7.2f}%")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
