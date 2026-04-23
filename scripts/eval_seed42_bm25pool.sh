#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
GPU=$1

echo "[$(date)] Evaluating seed42 BM25 model on BM25-only pool..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

echo "[$(date)] Evaluating seed42 Graph model on BM25-only pool (for matched comparison)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

echo "[$(date)] Done."
