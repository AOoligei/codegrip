#!/bin/bash
# Evaluate a RankFT model on both GREPO and SWE-bench Lite
# Usage: bash scripts/eval_rankft.sh <rankft_exp_dir> <gpu_id>
set -e
cd /home/chenlibin/grepo_agent

EXP_DIR=$1
GPU_ID=${2:-0}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python

if [ -z "$EXP_DIR" ]; then
    echo "Usage: $0 <rankft_exp_dir> <gpu_id>"
    exit 1
fi

# Find the best checkpoint
BEST_DIR="$EXP_DIR/best"
if [ ! -d "$BEST_DIR" ]; then
    # Try last checkpoint
    BEST_DIR=$(ls -td "$EXP_DIR"/checkpoint-* 2>/dev/null | head -1)
fi

if [ -z "$BEST_DIR" ] || [ ! -d "$BEST_DIR" ]; then
    echo "No checkpoint found in $EXP_DIR"
    exit 1
fi

MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
echo "=== Evaluating RankFT: $EXP_DIR ==="
echo "  Checkpoint: $BEST_DIR"
echo "  GPU: $GPU_ID"

# Eval on GREPO
echo ""
echo "--- GREPO Evaluation ---"
GREPO_OUT="$EXP_DIR/eval_grepo"
mkdir -p "$GREPO_OUT"
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir "$GREPO_OUT" \
    --gpu_id 0 \
    --top_k 200 2>&1 | tee "$GREPO_OUT/eval.log"

# Also eval with top_k=500
echo ""
echo "--- GREPO K=500 ---"
GREPO_OUT_500="$EXP_DIR/eval_grepo_k500"
mkdir -p "$GREPO_OUT_500"
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir "$GREPO_OUT_500" \
    --gpu_id 0 \
    --top_k 500 2>&1 | tee "$GREPO_OUT_500/eval.log"

# Eval on SWE-bench Lite
echo ""
echo "--- SWE-bench Lite Evaluation ---"
SWEBENCH_OUT="$EXP_DIR/eval_swebench"
mkdir -p "$SWEBENCH_OUT"
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir "$SWEBENCH_OUT" \
    --gpu_id 0 \
    --top_k 500 2>&1 | tee "$SWEBENCH_OUT/eval.log"

echo ""
echo "=== Evaluation Complete ==="
echo "GREPO K=200: $(cat $GREPO_OUT/summary.json 2>/dev/null)"
echo "GREPO K=500: $(cat $GREPO_OUT_500/summary.json 2>/dev/null)"
echo "SWE-bench:   $(cat $SWEBENCH_OUT/summary.json 2>/dev/null)"
