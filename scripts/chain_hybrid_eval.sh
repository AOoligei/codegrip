#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
SWEBENCH_PID=${1:-2333601}
GPU=0

echo "[$(date)] Waiting for SWE-bench eval (PID $SWEBENCH_PID) to finish..."
while kill -0 $SWEBENCH_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] GPU $GPU free. Starting hybrid retriever evals."

# Graph-hard reranker on hybrid+graph expanded pool
echo "[$(date)] Eval: graph-hard on hybrid+graph expanded pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_hybrid_matched_graph_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_hybrid_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_hybrid_graph/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on hybrid+graph: R@1=$R1"

# Graph-hard reranker on hybrid-only pool (no graph expansion)
echo "[$(date)] Eval: graph-hard on hybrid-only pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/hybrid_matched_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_hybrid_only \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_H=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_hybrid_only/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on hybrid-only: R@1=$R1_H"

# BM25-only reranker on hybrid+graph expanded pool
echo "[$(date)] Eval: BM25-only reranker on hybrid+graph expanded pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_hybrid_matched_graph_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_hybrid_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_BG=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runA_bm25only/eval_hybrid_graph/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] BM25-only on hybrid+graph: R@1=$R1_BG"

echo "[$(date)] === HYBRID RETRIEVER RESULTS ==="
echo "Graph-hard on hybrid+graph: R@1=$R1"
echo "Graph-hard on hybrid-only:  R@1=$R1_H"
echo "BM25-only on hybrid+graph:  R@1=$R1_BG"
echo "Compare: BM25+graph=27.01, BM25-only=19.00"
