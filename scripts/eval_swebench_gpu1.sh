#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
GPU=1

echo "[$(date)] SWE-bench evals on GPU $GPU"

# Graph-hard on SWE-bench expanded
echo "[$(date)] graph-hard + swebench expanded"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_merged_graph_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_swebench_expanded \
    --gpu_id 0 --top_k 200 --max_seq_length 512

# Graph-hard on SWE-bench BM25-only
echo "[$(date)] graph-hard + swebench bm25pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_only_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_swebench_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

# BM25-only reranker on SWE-bench expanded
echo "[$(date)] bm25-only + swebench expanded"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_merged_graph_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_swebench_expanded \
    --gpu_id 0 --top_k 200 --max_seq_length 512

# BM25-only reranker on SWE-bench BM25-only
echo "[$(date)] bm25-only + swebench bm25pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_only_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_swebench_bm25pool \
    --gpu_id 0 --top_k 200 --max_seq_length 512

echo "[$(date)] === SWE-BENCH ALL DONE ==="
for d in experiments/rankft_runB_graph/eval_swebench_expanded experiments/rankft_runB_graph/eval_swebench_bm25pool experiments/rankft_runA_bm25only/eval_swebench_expanded experiments/rankft_runA_bm25only/eval_swebench_bm25pool; do
    r=$(python3 -c "import json; print(f'{json.load(open(\"$d/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')" 2>/dev/null)
    echo "$d R@1=$r"
done
