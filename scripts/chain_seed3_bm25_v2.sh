#!/bin/bash
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
TRAIN_PID=3659435

echo "[$(date)] Waiting for seed3 bm25 training (PID $TRAIN_PID)..."
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; done

if [ ! -d "experiments/rankft_runA_bm25only_seed3/final" ]; then
    echo "[$(date)] ERROR: no final checkpoint"; exit 1
fi

echo "[$(date)] Running seed3 bm25 eval on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only_seed3/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only_seed3/eval_merged_rerank \
    --gpu_id 0 --top_k 200 --max_seq_length 512

echo "[$(date)] Seed3 bm25 eval done."
$PYTHON scripts/collect_seed_results.py
$PYTHON scripts/fill_paper_results.py
