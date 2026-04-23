#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent

# Wait for 3B bm25 training (PID 4005294) to finish
echo "[$(date)] Waiting for 3B bm25 training (PID 4005294)..."
while kill -0 4005294 2>/dev/null; do sleep 120; done
echo "[$(date)] 3B bm25 training done."

GPU=7

# Eval on expanded pool
echo "[$(date)] Evaluating 3B bm25 on expanded pool..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --lora_path experiments/scale_3B_bm25only/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_3B_bm25only/eval_merged_rerank \
    --gpu_id 0 --top_k 200 --max_seq_length 512

R1=$(python3 -c "import json; print(f'{json.load(open(\"experiments/scale_3B_bm25only/eval_merged_rerank/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] 3B bm25 R@1=$R1"
echo "[$(date)] Done. Fill this into paper scale table."
