#!/bin/bash
# Launch CodeGRIP evaluation
# Usage: bash scripts/launch_eval.sh <exp_name> <gpu_id> [eval_name] [test_data] [file_tree_dir]

set -e

EXP_NAME=$1
GPU_ID=$2
EVAL_NAME=${3:-eval_filetree}
TEST_DATA=${4:-data/grepo_text/grepo_test.jsonl}
FILE_TREE_DIR=${5:-data/file_trees}

EXP_DIR="experiments/$EXP_NAME"
LORA_PATH="$EXP_DIR/stage2_sft/final"
EVAL_DIR="$EXP_DIR/$EVAL_NAME"

if [ ! -d "$LORA_PATH" ]; then
    echo "Error: No final checkpoint at $LORA_PATH"
    exit 1
fi

# Get model path from config
CONFIG="$EXP_DIR/config.json"
MODEL_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['model_path'])")

mkdir -p "$EVAL_DIR"

echo "=== Evaluating $EXP_NAME on GPU $GPU_ID ==="
echo "  Model: $MODEL_PATH"
echo "  LoRA: $LORA_PATH"
echo "  Test data: $TEST_DATA"
echo "  Output: $EVAL_DIR"

nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID python src/eval/eval_grepo_file_level.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data $TEST_DATA \
    --file_tree_dir $FILE_TREE_DIR \
    --output_dir $EVAL_DIR \
    --prompt_mode filetree" > "$EVAL_DIR/eval.log" 2>&1 &
PID=$!
echo "  Launched PID: $PID on GPU $GPU_ID"
echo "  Log: $EVAL_DIR/eval.log"
echo "$PID" > "$EVAL_DIR/eval.pid"
