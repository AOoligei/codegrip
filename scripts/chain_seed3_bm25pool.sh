#!/bin/bash
# After seed3 bm25 expanded eval completes: run seed3 graph+bm25 on BM25 pool
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
CHAIN_PID=3660337  # chain_seed3_bm25_v2.sh
GPU=1

echo "[$(date)] Waiting for seed3 bm25 expanded eval chain (PID $CHAIN_PID)..."
while kill -0 $CHAIN_PID 2>/dev/null; do sleep 60; done

BM25_POOL=data/rankft/grepo_test_bm25_top500.jsonl
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct

# seed3 graph on BM25 pool
echo "[$(date)] Evaluating seed3 graph on BM25 pool..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL \
    --lora_path experiments/rankft_runB_graph_seed3/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates "$BM25_POOL" \
    --output_dir experiments/rankft_runB_graph_seed3/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] seed3 graph bm25pool done."

# seed3 bm25 on BM25 pool
echo "[$(date)] Evaluating seed3 bm25 on BM25 pool..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL \
    --lora_path experiments/rankft_runA_bm25only_seed3/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates "$BM25_POOL" \
    --output_dir experiments/rankft_runA_bm25only_seed3/eval_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] seed3 bm25 bm25pool done."

echo "[$(date)] All seed3 bm25pool evals complete."
