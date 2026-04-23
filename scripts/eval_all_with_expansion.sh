#!/bin/bash
# Evaluate all experiments with multi-signal expansion + reranking
# Usage: bash scripts/eval_all_with_expansion.sh [gpu_id]

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

MODEL_PATH_QWEN="/data/shuyang/models/Qwen2.5-7B-Instruct"
MODEL_PATH_CODER="/home/chenlibin/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
GPU="${1:-6}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval_full_pipeline() {
    local EXP_NAME=$1
    local MODEL_PATH=$2
    local ADAPTER_PATH=$3
    local EVAL_GPU=$4

    # Step 1: Run base evaluation
    if [ ! -f "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" ]; then
        if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
            echo "SKIP $EXP_NAME: adapter not found"
            return
        fi

        echo "=== Evaluating $EXP_NAME (base) on GPU $EVAL_GPU ==="
        CUDA_VISIBLE_DEVICES=$EVAL_GPU $PYTHON src/eval/eval_grepo_file_level.py \
            --model_path "$MODEL_PATH" \
            --lora_path "$ADAPTER_PATH" \
            --test_data data/grepo_text/grepo_test.jsonl \
            --output_dir "experiments/${EXP_NAME}/eval_filetree" \
            --prompt_mode filetree \
            --file_tree_dir data/file_trees 2>&1 | tee "experiments/${EXP_NAME}/eval_filetree.log"
    else
        echo "=== $EXP_NAME base eval already done ==="
    fi

    # Step 2: Run multi-signal expansion
    if [ -f "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" ] && \
       [ ! -f "experiments/${EXP_NAME}/eval_unified_expansion/predictions.jsonl" ]; then
        echo "=== Running expansion for $EXP_NAME ==="
        mkdir -p "experiments/${EXP_NAME}/eval_unified_expansion"
        $PYTHON src/eval/multi_signal_expansion.py \
            --predictions "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" \
            --train_data data/grepo_text/grepo_train.jsonl \
            --dep_graph_dir data/dep_graphs \
            --file_tree_dir data/file_trees \
            --output "experiments/${EXP_NAME}/eval_unified_expansion/predictions.jsonl" \
            --max_expand 35 --min_cochange_score 0.02 --max_dir_size 35 \
            2>&1 | tee "experiments/${EXP_NAME}/eval_expansion.log"
    fi

    # Step 3: Run issue-text-aware reranking
    if [ -f "experiments/${EXP_NAME}/eval_unified_expansion/predictions.jsonl" ] && \
       [ ! -f "experiments/${EXP_NAME}/eval_reranked/predictions.jsonl" ]; then
        echo "=== Running reranking for $EXP_NAME ==="
        mkdir -p "experiments/${EXP_NAME}/eval_reranked"
        $PYTHON src/eval/rerank_predictions.py \
            --predictions "experiments/${EXP_NAME}/eval_unified_expansion/predictions.jsonl" \
            --test_data data/grepo_text/grepo_test.jsonl \
            --output "experiments/${EXP_NAME}/eval_reranked/predictions.jsonl" \
            --promote_top5 1 --promote_top10 3 --threshold 0.2 \
            2>&1 | tee "experiments/${EXP_NAME}/eval_reranking.log"
    fi

    echo "=== Done: $EXP_NAME ==="
    echo
}

# Evaluate all experiments
echo "============================================"
echo "=== Full Evaluation Pipeline ==="
echo "============================================"

# Exp 1: SFT only (Qwen2.5-7B)
eval_full_pipeline "exp1_sft_only" "$MODEL_PATH_QWEN" "experiments/exp1_sft_only/stage2_sft/final" "$GPU"

# Exp 5: Coder SFT
eval_full_pipeline "exp5_coder_sft_only" "$MODEL_PATH_CODER" "experiments/exp5_coder_sft_only/stage2_sft/final" "$GPU"

# Exp 6: Warm-start
eval_full_pipeline "exp6_warmstart_cochange" "$MODEL_PATH_QWEN" "experiments/exp6_warmstart_cochange/stage2_sft/final" "$GPU"

# Exp 7: Multi-task
eval_full_pipeline "exp7_multitask_sft" "$MODEL_PATH_QWEN" "experiments/exp7_multitask_sft/stage2_sft/final" "$GPU"

# Print comprehensive summary
echo
echo "============================================"
echo "=== COMPREHENSIVE RESULTS SUMMARY ==="
echo "============================================"
echo
echo "Method                          | Hit@1  | Hit@5  | Hit@10 | Hit@20"
echo "--------------------------------|--------|--------|--------|-------"

for d in exp1_sft_only exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
    # Base results
    if [ -f "experiments/$d/eval_filetree/summary.json" ]; then
        echo -n "$(printf '%-32s' "$d (base)")"
        $PYTHON -c "
import json
with open('experiments/$d/eval_filetree/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"| {o.get('hit@1',0):6.2f} | {o.get('hit@5',0):6.2f} | {o.get('hit@10',0):6.2f} | {o.get('hit@20',0):6.2f}\")
"
    fi
    # Expanded results
    if [ -f "experiments/$d/eval_unified_expansion/summary.json" ]; then
        echo -n "$(printf '%-32s' "$d (+expand)")"
        $PYTHON -c "
import json
with open('experiments/$d/eval_unified_expansion/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"| {o.get('hit@1',0):6.2f} | {o.get('hit@5',0):6.2f} | {o.get('hit@10',0):6.2f} | {o.get('hit@20',0):6.2f}\")
"
    fi
    # Reranked results
    if [ -f "experiments/$d/eval_reranked/summary.json" ]; then
        echo -n "$(printf '%-32s' "$d (+rerank)")"
        $PYTHON -c "
import json
with open('experiments/$d/eval_reranked/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"| {o.get('hit@1',0):6.2f} | {o.get('hit@5',0):6.2f} | {o.get('hit@10',0):6.2f} | {o.get('hit@20',0):6.2f}\")
"
    fi
done

echo "--------------------------------|--------|--------|--------|-------"
echo "GAT (GREPO paper)               | 14.80  | 31.51  | 37.40  | 41.25"
echo "Agentless (paper)               | 13.65  | 21.86  | 23.43  | 23.43"
