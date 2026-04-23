#!/bin/bash
# Ablation: same as codeaware_swetrain but path_augment_fraction=0 + --include_code.
# Holds constant: training data, hyperparams, negative mix, file_tree/dep_graph paths.
# Tests: does the path_augment-shuffle component contribute, or is "code in prompt
# + SWE-bench train" alone responsible for codeaware's gain?
GPU=${1:-7}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/home/chenlibin/grepo_agent/experiments/rankft_codeonly_swetrain
cd /home/chenlibin/grepo_agent
CUDA_VISIBLE_DEVICES=$GPU $PY -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/swebench_train/swebench_train.jsonl \
    --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
    --file_tree_dir data/swebench_lite/file_trees \
    --dep_graph_dir data/swebench_lite/dep_graphs \
    --output_dir $OUT \
    --device cuda:0 \
    --num_negatives 8 \
    --neg_bm25_ratio 0.25 --neg_graph_ratio 0.1 --neg_samedir_ratio 0.5 --neg_random_ratio 0.15 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 50 \
    --logging_steps 10 \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --seed 42 \
    --path_augment_fraction 0.0 \
    --include_code \
    --code_max_lines 50 \
    --repo_dir data/swebench_lite/repos
