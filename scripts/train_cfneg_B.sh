#!/bin/bash
# E4: Counterfactual-negative training Variant B
# Negative mix: bm25=0.25 graph=0.25 treeneighbor=0.25 pathdist=0.25
# Hypothesis: tree-sibling confusion is a better hard-negative source

GPU_ID=${1:-1}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3

echo "=== CF-Neg Variant B Training (GPU $GPU_ID) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --file_tree_dir data/file_trees \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --output_dir experiments/rankft_cfneg_B \
    --device cuda:0 \
    --num_negatives 16 \
    --neg_bm25_ratio 0.25 \
    --neg_graph_ratio 0.25 \
    --neg_treeneighbor_ratio 0.25 \
    --neg_pathdist_ratio 0.25 \
    --neg_random_ratio 0.0 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank 32 \
    --seed 42

echo "End: $(date)"
