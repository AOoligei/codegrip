#!/bin/bash
# Code-residual ablation: 100 lines instead of 30 (fairer code representation)
# Compare with v2 (30 lines) to see if more code context helps

set -euo pipefail

GPU=${1:-4}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/data/chenlibin/grepo_agent_experiments/code_residual_7b_100lines

mkdir -p $OUT

echo "Starting code-residual 100-line ablation on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/train/train_rankft_code_residual.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/rankft/grepo_train.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir $OUT \
    --device cuda:0 \
    --num_negatives 4 \
    --max_seq_length 1024 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --save_steps 50 \
    --code_max_lines 100 \
    --neg_samedir_ratio 0.25 --neg_pathdist_ratio 0.25 --neg_bm25_ratio 0.25 --neg_random_ratio 0.25 \
    --neg_graph_ratio 0.0 \
    --anonymize_paths \
    2>&1 | tee $OUT/train.log
