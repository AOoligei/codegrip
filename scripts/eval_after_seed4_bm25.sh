#!/bin/bash
# Wait for seed4 bm25 training on GPU 5 to finish, then eval
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
TRAIN_PID=2019885

echo "[$(date)] Waiting for seed4 bm25 training (PID $TRAIN_PID) to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] Seed4 bm25 training done."

if [ ! -d "experiments/rankft_runA_bm25only_seed4/final" ]; then
    echo "[$(date)] ERROR: no final checkpoint found"
    exit 1
fi

echo "[$(date)] Starting seed4 bm25 eval on GPU 5..."
CUDA_VISIBLE_DEVICES=5 $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only_seed4/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/rankft_runA_bm25only_seed4/eval_merged_rerank \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee experiments/rankft_runA_bm25only_seed4_eval.log

echo "[$(date)] Seed4 bm25 eval done."

# Collect all seed results
echo "[$(date)] Collecting seed results..."
$PYTHON scripts/collect_seed_results.py
