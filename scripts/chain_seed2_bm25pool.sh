#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
GPU=6
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
BM25_POOL=data/rankft/grepo_test_bm25_top500.jsonl

echo "[$(date)] seed2 graph bm25pool eval on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL --lora_path experiments/rankft_runB_graph_seed2/final \
    --test_data data/grepo_text/grepo_test.jsonl --bm25_candidates "$BM25_POOL" \
    --output_dir experiments/rankft_runB_graph_seed2/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] seed2 graph bm25pool DONE"

echo "[$(date)] seed2 bm25 bm25pool eval on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL --lora_path experiments/rankft_runA_bm25only_seed2/final \
    --test_data data/grepo_text/grepo_test.jsonl --bm25_candidates "$BM25_POOL" \
    --output_dir experiments/rankft_runA_bm25only_seed2/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] seed2 bm25 bm25pool DONE"

echo "[$(date)] All seed2 bm25pool evals complete!"
