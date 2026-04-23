#!/bin/bash
# Train SWE-bench reranker with delex50 (mitigation experiment)
# Usage: bash scripts/swebench_train_delex50.sh [GPU_ID]

GPU_ID=${1:-4}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

echo "=== SWE-bench Delex50 Training (GPU $GPU_ID) ==="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/swebench_train/swebench_train.jsonl \
    --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
    --file_tree_dir data/swebench_lite/file_trees \
    --output_dir experiments/swebench_rankft_delex50 \
    --device cuda:0 \
    --num_negatives 16 \
    --neg_bm25_ratio 0.75 \
    --neg_graph_ratio 0.0 \
    --neg_random_ratio 0.25 \
    --num_epochs 2 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --save_steps 100 \
    --logging_steps 10 \
    --seed 42 \
    --delex_fraction 0.5
