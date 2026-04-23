#!/bin/bash
# Evaluate SWE-bench adapted model on SWE-bench Lite.
# Run multiple evaluations in parallel on available GPUs.
#
# Usage: bash scripts/eval_adapted_swebench.sh [GPU_LIST]
# Example: bash scripts/eval_adapted_swebench.sh "0,3"

set -e

ADAPTED_MODEL="experiments/rankft_swebench_adapted/best"
if [ ! -d "$ADAPTED_MODEL" ]; then
    ADAPTED_MODEL="experiments/rankft_swebench_adapted/final"
fi

if [ ! -f "$ADAPTED_MODEL/adapter_model.safetensors" ]; then
    echo "Error: No model checkpoint found at $ADAPTED_MODEL"
    exit 1
fi

echo "=== Evaluating adapted model: $ADAPTED_MODEL ==="

# Default to GPUs 0 and 3
GPU_LIST="${1:-0,3}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"

# Evaluation configurations
declare -A EVALS=(
    ["best_ensemble_512"]="data/rankft/swebench_bm25_final_top500.jsonl 512 16"
    ["best_ensemble_1024"]="data/rankft/swebench_bm25_final_top500.jsonl 1024 8"
)

IDX=0
PIDS=()
for eval_name in "${!EVALS[@]}"; do
    IFS=' ' read -r bm25_file max_seq bs <<< "${EVALS[$eval_name]}"
    GPU_IDX=$((IDX % ${#GPUS[@]}))
    GPU=${GPUS[$GPU_IDX]}

    OUTPUT_DIR="experiments/rankft_swebench_adapted/eval_swebench_${eval_name}"

    echo "  Launching $eval_name on GPU $GPU (max_seq=$max_seq, bs=$bs)"
    CUDA_VISIBLE_DEVICES=$GPU python -u src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path "$ADAPTED_MODEL" \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --bm25_candidates "$bm25_file" \
        --max_seq_length "$max_seq" \
        --score_batch_size "$bs" \
        --top_k 50 \
        --output_dir "$OUTPUT_DIR" \
        > "logs/eval_adapted_${eval_name}.log" 2>&1 &
    PIDS+=($!)
    IDX=$((IDX + 1))
done

echo "  Launched ${#PIDS[@]} evaluations. PIDs: ${PIDS[*]}"
echo "  Monitor with: tail -f logs/eval_adapted_*.log"
