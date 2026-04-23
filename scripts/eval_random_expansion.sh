#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent

GPU=${1:-0}
echo "[$(date)] Starting random expansion control eval on GPU $GPU"

# Graph-hard reranker on random-expanded pool
echo "[$(date)] Eval: graph-hard reranker on random expansion pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_random_expansion_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_random_expansion \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512

R1_GRAPH=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_random_expansion/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on random expansion: R@1=$R1_GRAPH"

# BM25-only reranker on random-expanded pool
echo "[$(date)] Eval: BM25-only reranker on random expansion pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_random_expansion_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_random_expansion \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512

R1_BM25=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runA_bm25only/eval_random_expansion/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] BM25-only on random expansion: R@1=$R1_BM25"

echo "[$(date)] === RANDOM EXPANSION CONTROL RESULTS ==="
echo "Graph-hard on random expansion: R@1=$R1_GRAPH"
echo "BM25-only on random expansion: R@1=$R1_BM25"
echo "Compare with graph expansion: Graph=27.01, BM25=23.83"
echo "Compare with BM25-only pool: Graph=19.00, BM25=18.35"

# Size-matched BM25 control (same pool size, just BM25-top-K)
echo "[$(date)] Eval: graph-hard reranker on size-matched BM25 pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/bm25_top_matched_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_bm25_matched \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512

R1_MATCHED=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_bm25_matched/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on size-matched BM25: R@1=$R1_MATCHED"

echo "[$(date)] === ALL POOL COMPARISON ==="
echo "BM25-only pool (top-200): R@1=19.00"
echo "Size-matched BM25 (top-K~208): R@1=$R1_MATCHED"
echo "Random expansion (~208): R@1=$R1_GRAPH"
echo "Graph expansion (~208): R@1=27.01"
