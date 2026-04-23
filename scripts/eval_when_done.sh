#!/bin/bash
# Wait for experiments to finish, then evaluate
# Usage: nohup bash scripts/eval_when_done.sh > scripts/eval_when_done.log 2>&1 &

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

MODEL_PATH_QWEN="/data/shuyang/models/Qwen2.5-7B-Instruct"
MODEL_PATH_CODER="/home/chenlibin/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval_exp() {
    local EXP_NAME=$1
    local MODEL_PATH=$2
    local ADAPTER_PATH=$3
    local GPU=$4

    if [ -f "experiments/${EXP_NAME}/eval_filetree/summary.json" ]; then
        echo "SKIP $EXP_NAME: already evaluated"
        return
    fi

    echo "=== Evaluating $EXP_NAME on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$ADAPTER_PATH" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir "experiments/${EXP_NAME}/eval_filetree" \
        --prompt_mode filetree \
        --file_tree_dir data/file_trees 2>&1 | tee "experiments/${EXP_NAME}/eval_filetree.log"

    # Also run co-change expansion
    if [ -f "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" ]; then
        mkdir -p "experiments/${EXP_NAME}/eval_filetree_expanded"
        $PYTHON src/eval/cochange_expansion.py \
            --predictions "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" \
            --train_data data/grepo_text/grepo_train.jsonl \
            --output "experiments/${EXP_NAME}/eval_filetree_expanded/predictions.jsonl" \
            --min_cochange 1 \
            --min_score 0.05 \
            --max_expand 20 2>&1 | tee "experiments/${EXP_NAME}/eval_expanded.log"
    fi
}

# Wait for each experiment and evaluate
wait_and_eval() {
    local EXP_NAME=$1
    local MODEL_PATH=$2
    local ADAPTER_PATH=$3
    local GPU=$4

    echo "Waiting for $EXP_NAME..."
    while [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; do
        sleep 60
    done
    echo "$EXP_NAME training complete!"
    eval_exp "$EXP_NAME" "$MODEL_PATH" "$ADAPTER_PATH" "$GPU"
}

# Run evaluations as experiments complete (use GPUs 6,7 for eval)
wait_and_eval "exp5_coder_sft_only" "$MODEL_PATH_CODER" "experiments/exp5_coder_sft_only/stage2_sft/final" 6 &
PID5=$!

wait_and_eval "exp6_warmstart_cochange" "$MODEL_PATH_QWEN" "experiments/exp6_warmstart_cochange/stage2_sft/final" 7 &
PID6=$!

# Wait for first two to finish eval, then use GPU 6 for exp7
wait $PID5 $PID6
wait_and_eval "exp7_multitask_sft" "$MODEL_PATH_QWEN" "experiments/exp7_multitask_sft/stage2_sft/final" 6

# Print final summary
echo
echo "=========================================="
echo "=== FINAL RESULTS SUMMARY ==="
echo "=========================================="
for d in exp1_sft_only exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
    echo "--- $d ---"
    if [ -f "experiments/$d/eval_filetree/summary.json" ]; then
        echo -n "  Base: "
        $PYTHON -c "
import json
with open('experiments/$d/eval_filetree/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"Hit@1={o.get('hit@1',0):.2f}% Hit@5={o.get('hit@5',0):.2f}% Hit@10={o.get('hit@10',0):.2f}% Hit@20={o.get('hit@20',0):.2f}%\")
"
    fi
    if [ -f "experiments/$d/eval_filetree_expanded/summary.json" ]; then
        echo -n "  +Exp: "
        $PYTHON -c "
import json
with open('experiments/$d/eval_filetree_expanded/summary.json') as f:
    s = json.load(f)
o = s['overall']
print(f\"Hit@1={o.get('hit@1',0):.2f}% Hit@5={o.get('hit@5',0):.2f}% Hit@10={o.get('hit@10',0):.2f}% Hit@20={o.get('hit@20',0):.2f}%\")
"
    fi
done
