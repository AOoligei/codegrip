#!/bin/bash
# Evaluate all trained experiments on GREPO test set
# Usage: bash scripts/run_eval_all.sh [gpu_id]

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
GPU="${1:-2}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_eval() {
    local EXP_NAME=$1
    local ADAPTER_PATH=$2
    local EVAL_GPU=$3

    if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
        echo "SKIP $EXP_NAME: adapter not found at $ADAPTER_PATH"
        return
    fi

    echo "=== Evaluating $EXP_NAME on GPU $EVAL_GPU ==="
    CUDA_VISIBLE_DEVICES=$EVAL_GPU $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$ADAPTER_PATH" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir "experiments/${EXP_NAME}/eval_filetree" \
        --prompt_mode filetree \
        --file_tree_dir data/file_trees 2>&1 | tee "experiments/${EXP_NAME}/eval_filetree.log"

    echo "=== Done: $EXP_NAME ==="
    echo
}

# Evaluate each experiment
run_eval "exp1_sft_only" "experiments/exp1_sft_only/stage2_sft/final" "$GPU"
run_eval "exp2_cochange_gsp_sft" "experiments/exp2_cochange_gsp_sft/stage2_sft/final" "$GPU"
run_eval "exp3_ast_gsp_sft" "experiments/exp3_ast_gsp_sft/stage2_sft/final" "$GPU"
run_eval "exp4_combined_gsp_sft" "experiments/exp4_combined_gsp_sft/stage2_sft/final" "$GPU"

echo "=== All evaluations complete ==="

# Print summary
echo
echo "=== RESULTS SUMMARY ==="
for d in exp1_sft_only exp2_cochange_gsp_sft exp3_ast_gsp_sft exp4_combined_gsp_sft; do
    if [ -f "experiments/$d/eval_filetree/summary.json" ]; then
        echo "--- $d ---"
        python -c "
import json
with open('experiments/$d/eval_filetree/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"  Hit@1={o.get('hit@1',0):.2f}% Hit@3={o.get('hit@3',0):.2f}% Hit@5={o.get('hit@5',0):.2f}% Hit@10={o.get('hit@10',0):.2f}% Hit@20={o.get('hit@20',0):.2f}%\")
"
    fi
done
