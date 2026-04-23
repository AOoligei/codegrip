#!/bin/bash
# Eval a model on PathSwap-GREPO (graph pool, pathswapped paths)
# Usage: bash scripts/eval_pathswap.sh <GPU_ID> <LORA_PATH> <OUTPUT_NAME>
GPU_ID=${1:-5}
LORA_PATH=${2:-experiments/rankft_runB_graph/best}
OUTPUT_NAME=${3:-7B_graph}

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL_PATH=/data/shuyang/models/Qwen2.5-7B-Instruct

echo "=== PathSwap Eval: $OUTPUT_NAME (GPU $GPU_ID) ==="
echo "LoRA: $LORA_PATH"
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data data/pathswap/grepo_test_pathswap.jsonl \
    --bm25_candidates data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl \
    --output_dir experiments/pathswap_eval/${OUTPUT_NAME}_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "Graph pool done: $(date)"

# Also eval on hybrid pool
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data data/pathswap/grepo_test_pathswap.jsonl \
    --bm25_candidates data/pathswap/merged_hybrid_e5large_graph_candidates_pathswap.jsonl \
    --output_dir experiments/pathswap_eval/${OUTPUT_NAME}_hybrid \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "Hybrid pool done: $(date)"
echo "=== Summary ==="
for pool in graph hybrid; do
    f="experiments/pathswap_eval/${OUTPUT_NAME}_${pool}/summary.json"
    if [ -f "$f" ]; then
        $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${OUTPUT_NAME}_${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
    fi
done
echo "End: $(date)"
