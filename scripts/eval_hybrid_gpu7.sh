#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
GPU=7

echo "[$(date)] Hybrid retriever evals on GPU $GPU"

# Graph-hard on hybrid+graph expanded
echo "[$(date)] graph-hard + hybrid+graph"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_hybrid_matched_graph_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_hybrid_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512

# Graph-hard on hybrid-only (no graph expansion)
echo "[$(date)] graph-hard + hybrid-only"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/hybrid_matched_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_hybrid_only \
    --gpu_id 0 --top_k 200 --max_seq_length 512

# BM25-only reranker on hybrid+graph
echo "[$(date)] bm25-only + hybrid+graph"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_hybrid_matched_graph_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_hybrid_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512

echo "[$(date)] === HYBRID ALL DONE ==="
for d in experiments/rankft_runB_graph/eval_hybrid_graph experiments/rankft_runB_graph/eval_hybrid_only experiments/rankft_runA_bm25only/eval_hybrid_graph; do
    r=$(python3 -c "import json; print(f'{json.load(open(\"$d/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')" 2>/dev/null)
    echo "$d R@1=$r"
done
