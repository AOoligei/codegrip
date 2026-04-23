#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
EDGE_PID=${1:-2316449}
GPU=0

echo "[$(date)] Waiting for edge-type ablation (PID $EDGE_PID) to finish..."
while kill -0 $EDGE_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] GPU $GPU free. Starting SWE-bench evals."

# Graph-hard reranker on SWE-bench expanded pool
echo "[$(date)] Eval: graph-hard reranker on SWE-bench expanded pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_merged_graph_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_swebench_expanded \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_GRAPH_EXP=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_swebench_expanded/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on SWE-bench expanded: R@1=$R1_GRAPH_EXP"

# Graph-hard reranker on SWE-bench BM25-only pool
echo "[$(date)] Eval: graph-hard reranker on SWE-bench BM25-only pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_only_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_swebench_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_GRAPH_BM25=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_swebench_bm25pool/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Graph-hard on SWE-bench BM25-only: R@1=$R1_GRAPH_BM25"

# BM25-only reranker on SWE-bench expanded pool
echo "[$(date)] Eval: BM25-only reranker on SWE-bench expanded pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_merged_graph_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_swebench_expanded \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_BM25_EXP=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runA_bm25only/eval_swebench_expanded/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] BM25-only on SWE-bench expanded: R@1=$R1_BM25_EXP"

# BM25-only reranker on SWE-bench BM25-only pool
echo "[$(date)] Eval: BM25-only reranker on SWE-bench BM25-only pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_only_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_swebench_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_BM25_BM25=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runA_bm25only/eval_swebench_bm25pool/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] BM25-only on SWE-bench BM25-only: R@1=$R1_BM25_BM25"

echo "[$(date)] === SWE-BENCH RESULTS ==="
echo "              | BM25 pool | Expanded pool"
echo "Graph-hard    | $R1_GRAPH_BM25     | $R1_GRAPH_EXP"
echo "BM25-only     | $R1_BM25_BM25     | $R1_BM25_EXP"
