#!/bin/bash
GPU=${1:-5}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/home/chenlibin/grepo_agent/experiments/rankft_consistency_swe_7b
cd /home/chenlibin/grepo_agent
CUDA_VISIBLE_DEVICES=$GPU $PY -u src/train/train_rankft_consistency.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/swebench_train/swebench_train.jsonl \
    --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
    --file_tree_dir data/file_trees \
    --dep_graph_dir data/dep_graphs \
    --output_dir $OUT \
    --device cuda:0 \
    --num_negatives 8 \
    --neg_bm25_ratio 0.75 --neg_graph_ratio 0.0 --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 50 \
    --logging_steps 10 \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --seed 42 \
    --code_max_lines 50 \
    --repo_dir data/swebench_lite/repos \
    --consistency_lambda 1.0 \
    --hash_ce_weight 0.5
