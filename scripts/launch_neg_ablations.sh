#!/bin/bash
# Negative mining ablation experiments
# Run each on a separate GPU after current training completes
# Each takes ~3h on a single 4090

BASE_CMD="/home/chenlibin/miniconda3/envs/tgn/bin/python src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --num_negatives 16 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank 32"

# Ablation 1: No graph negatives (BM25+Random only)
echo "=== Ablation: No graph negatives ==="
echo "CUDA_VISIBLE_DEVICES=\$GPU $BASE_CMD \\
    --neg_bm25_ratio 0.75 --neg_graph_ratio 0.0 --neg_random_ratio 0.25 \\
    --output_dir experiments/ablation_no_graph_neg \\
    --device cuda:0"
echo

# Ablation 2: No BM25 negatives (Graph+Random only)
echo "=== Ablation: No BM25 negatives ==="
echo "CUDA_VISIBLE_DEVICES=\$GPU $BASE_CMD \\
    --neg_bm25_ratio 0.0 --neg_graph_ratio 0.5 --neg_random_ratio 0.5 \\
    --output_dir experiments/ablation_no_bm25_neg \\
    --device cuda:0"
echo

# Ablation 3: Random negatives only (no structured mining)
echo "=== Ablation: Random negatives only ==="
echo "CUDA_VISIBLE_DEVICES=\$GPU $BASE_CMD \\
    --neg_bm25_ratio 0.0 --neg_graph_ratio 0.0 --neg_random_ratio 1.0 \\
    --output_dir experiments/ablation_random_neg \\
    --device cuda:0"
echo

echo "Each ablation takes ~3 hours on a single GPU."
echo "Run on GPUs that become free (check nvidia-smi)."
echo "After training, evaluate with: python src/eval/eval_rankft.py --lora_path <ablation>/best --candidates data/rankft/merged_bm25_exp6_candidates.jsonl"
