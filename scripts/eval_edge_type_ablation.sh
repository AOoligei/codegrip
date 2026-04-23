#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
GPU=${1:-0}

echo "[$(date)] Starting edge-type ablation evals on GPU $GPU"

# Co-change only expansion
echo "[$(date)] Eval: graph-hard reranker on co-change-only pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_cochange_only_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_cochange_only \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_CC=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_cochange_only/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Co-change only: R@1=$R1_CC"

# Import only expansion
echo "[$(date)] Eval: graph-hard reranker on import-only pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_import_only_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_import_only \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_IMP=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_import_only/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Import only: R@1=$R1_IMP"

# Both edge types (reproduced)
echo "[$(date)] Eval: graph-hard reranker on both-edge-types pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_both_edge_types_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_both_edge_types \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1_BOTH=$(python3 -c "import json; print(f'{json.load(open(\"experiments/rankft_runB_graph/eval_both_edge_types/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Both edge types: R@1=$R1_BOTH"

echo "[$(date)] === EDGE-TYPE ABLATION RESULTS ==="
echo "Co-change only: R@1=$R1_CC"
echo "Import only:    R@1=$R1_IMP"
echo "Both (reprod):  R@1=$R1_BOTH"
echo "Original graph: R@1=27.01"
