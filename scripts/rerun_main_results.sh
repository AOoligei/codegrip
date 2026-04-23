#!/bin/bash
# Rerun all main results with updated eval script (per_repo + CI + strict Acc@k)
# After running, use scripts/collect_main_results.py to generate paper tables
#
# Prerequisites:
#   - Updated eval_rankft_4bit.py with per_repo, bootstrap CI, strict Acc@k
#   - Validation split: data/grepo_text/grepo_val.jsonl
#   - Val BM25 candidates: data/rankft/grepo_val_bm25_top500.jsonl

set -euo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
QWEN=/data/shuyang/models/Qwen2.5-7B-Instruct
LLAMA=/data/hzy/models/Llama-3.1-8B-Instruct
QWEN3=/data/hzy/models/Qwen3-8B
OUT_BASE=/data/chenlibin/grepo_agent_experiments/v2_with_ci

# ============================================================
# 1. Main baselines (graph-expanded pool)
# ============================================================
run_eval() {
    local gpu=$1 model=$2 lora=$3 test=$4 cands=$5 outdir=$6
    echo "GPU $gpu: $outdir"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/eval_rankft_4bit.py \
        --model_path "$model" \
        --lora_path "$lora" \
        --test_data "$test" \
        --bm25_candidates "$cands" \
        --output_dir "$OUT_BASE/$outdir" \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 4
}

TEST=data/grepo_text/grepo_test.jsonl
CANDS=data/rankft/merged_bm25_exp6_candidates.jsonl

# Qwen2.5-7B baselines
run_eval 0 $QWEN experiments/rankft_runB_graph/best $TEST $CANDS qwen25_7b/eval_graph &
run_eval 4 $QWEN experiments/rankft_runB_graph/final $TEST $CANDS qwen25_7b/eval_graph_final &

# Cross-LLM baselines
run_eval 5 $LLAMA experiments/cross_llm_llama31_8b/best $TEST $CANDS llama31_8b/eval_graph &
run_eval 6 $QWEN3 /data/chenlibin/grepo_agent_experiments/cross_llm_qwen3_8b/best $TEST $CANDS qwen3_8b/eval_graph &

wait
echo "=== Baselines done ==="

# ============================================================
# 2. Perturbation evals (all 3 models x 6 conditions)
# ============================================================
PERTURBS="shuffle_filenames shuffle_dirs flatten_dirs swap_leaf_dirs remove_module_names delexicalize"

for cond in $PERTURBS; do
    PTEST=experiments/path_perturb_${cond}/test.jsonl
    [ ! -f "$PTEST" ] && echo "SKIP: $PTEST not found" && continue

    # Qwen2.5
    run_eval 0 $QWEN experiments/rankft_runB_graph/best $PTEST $CANDS qwen25_7b/eval_perturb_${cond} &
    # Llama
    run_eval 4 $LLAMA experiments/cross_llm_llama31_8b/best $PTEST $CANDS llama31_8b/eval_perturb_${cond} &
    # Qwen3
    run_eval 5 $QWEN3 /data/chenlibin/grepo_agent_experiments/cross_llm_qwen3_8b/best $PTEST $CANDS qwen3_8b/eval_perturb_${cond} &

    wait
done
echo "=== Perturbations done ==="

# ============================================================
# 3. Code-residual model
# ============================================================
run_eval 0 $QWEN experiments/code_residual_7b/best $TEST $CANDS code_residual_7b/eval_graph &
for cond in $PERTURBS; do
    PTEST=experiments/path_perturb_${cond}/test.jsonl
    [ ! -f "$PTEST" ] && continue
    run_eval 4 $QWEN experiments/code_residual_7b/best $PTEST $CANDS code_residual_7b/eval_perturb_${cond} &
    wait
done
echo "=== Code-residual done ==="

# ============================================================
# 4. Validation set eval (for alpha selection)
# ============================================================
VAL_TEST=data/grepo_text/grepo_val.jsonl
VAL_CANDS=data/rankft/grepo_val_bm25_top500.jsonl

run_eval 0 $QWEN experiments/rankft_runB_graph/best $VAL_TEST $VAL_CANDS qwen25_7b/eval_val_graph &
run_eval 4 $QWEN experiments/code_residual_7b/best $VAL_TEST $VAL_CANDS code_residual_7b/eval_val_graph &
wait
echo "=== Validation evals done ==="

echo ""
echo "ALL RERUNS COMPLETE"
echo "Run: python scripts/collect_main_results.py to generate tables"
