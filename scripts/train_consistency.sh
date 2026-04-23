#!/bin/bash
GPU=${1:-5}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/home/chenlibin/grepo_agent/experiments/rankft_consistency_7b
cd /home/chenlibin/grepo_agent
CUDA_VISIBLE_DEVICES=$GPU $PY -u src/train/train_rankft_consistency.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --file_tree_dir data/file_trees \
    --dep_graph_dir data/dep_graphs \
    --output_dir $OUT \
    --device cuda:0 \
    --num_negatives 8 \
    --neg_bm25_ratio 0.5 --neg_graph_ratio 0.25 --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --logging_steps 20 \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --seed 42 \
    --code_max_lines 50 \
    --repo_dir data/repos \
    --consistency_lambda 1.0 \
    --hash_ce_weight 0.5
