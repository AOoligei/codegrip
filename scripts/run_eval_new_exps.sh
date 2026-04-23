#!/bin/bash
# Evaluate new experiments (exp5-exp7) on GREPO test set
# Usage: bash scripts/run_eval_new_exps.sh [gpu_id]

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

MODEL_PATH_QWEN="/data/shuyang/models/Qwen2.5-7B-Instruct"
MODEL_PATH_CODER="/home/chenlibin/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
GPU="${1:-6}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_eval() {
    local EXP_NAME=$1
    local MODEL_PATH=$2
    local ADAPTER_PATH=$3
    local EVAL_GPU=$4

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

# Exp 5: Qwen2.5-Coder-7B-Instruct SFT
run_eval "exp5_coder_sft_only" "$MODEL_PATH_CODER" "experiments/exp5_coder_sft_only/stage2_sft/final" "$GPU"

# Exp 6: Warm-start from co-change GSP
run_eval "exp6_warmstart_cochange" "$MODEL_PATH_QWEN" "experiments/exp6_warmstart_cochange/stage2_sft/final" "$GPU"

# Exp 7: Multi-task SFT (80% SFT + 20% GSP)
run_eval "exp7_multitask_sft" "$MODEL_PATH_QWEN" "experiments/exp7_multitask_sft/stage2_sft/final" "$GPU"

echo "=== All evaluations complete ==="

# Print summary
echo
echo "=== RESULTS SUMMARY ==="
for d in exp1_sft_only exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
    if [ -f "experiments/$d/eval_filetree/summary.json" ]; then
        echo "--- $d ---"
        $PYTHON -c "
import json
with open('experiments/$d/eval_filetree/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"  Hit@1={o.get('hit@1',0):.2f}% Hit@3={o.get('hit@3',0):.2f}% Hit@5={o.get('hit@5',0):.2f}% Hit@10={o.get('hit@10',0):.2f}% Hit@20={o.get('hit@20',0):.2f}%\")
"
    fi
done
