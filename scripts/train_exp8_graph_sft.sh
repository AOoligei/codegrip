#!/bin/bash
# Exp 8: Graph-conditioned SFT
# Train on v3_graph data (structural context in prompts)
# Uses GPU 4 (or first available GPU after exp5/6/7 finish)
#
# Usage: bash scripts/train_exp8_graph_sft.sh [gpu_id]

set -e

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

GPU="${1:-4}"
MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
EXP_NAME="exp8_graph_sft"
SFT_DATA="data/sft/sft_v3_graph.jsonl"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "=== Exp 8: Graph-Conditioned SFT ==="
echo "=== GPU: $GPU ==="
echo "=== Data: $SFT_DATA ==="
echo "============================================"

# Verify data exists
if [ ! -f "$SFT_DATA" ]; then
    echo "Generating v3_graph SFT data..."
    $PYTHON src/data/generate_graph_sft_data.py --output_dir data/sft
fi

wc -l "$SFT_DATA"

# Train
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/train/train_codegrip.py \
    --model_path "$MODEL_PATH" \
    --sft_data "$SFT_DATA" \
    --output_dir "experiments/$EXP_NAME" \
    --skip_gsp \
    --sft_epochs 3 \
    --sft_lr 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 3072 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --save_steps 200 \
    --logging_steps 10 \
    --device "cuda:0" \
    2>&1 | tee "experiments/${EXP_NAME}.log"

echo "=== Training complete ==="

# Auto-evaluate
echo "=== Running evaluation ==="
bash scripts/eval_exp8.sh "$GPU"
