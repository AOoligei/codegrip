#!/bin/bash
# Train path-debiased reranker: 50% normal + 50% delexicalized prompts
# Bold experiment: force the model to learn non-lexical signals
# Uses same config as rankft_runB_graph but with --delex_fraction 0.5

GPU_ID=${1:-3}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3

echo "=== Delex Reranker Training (GPU $GPU_ID) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --output_dir experiments/rankft_delex50 \
    --device cuda:0 \
    --num_negatives 16 \
    --neg_bm25_ratio 0.5 \
    --neg_graph_ratio 0.25 \
    --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank 32 \
    --seed 42 \
    --delex_fraction 0.5

echo "End: $(date)"
