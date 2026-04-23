#!/bin/bash
# GPU 3 pipeline: after seed1 bm25only training → eval → 1.5B/3B scale ablation (if models downloaded)
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [GPU3] $1" | tee -a orchestrate.log; }

# Wait for seed1 bm25only training (PID 2890806) to finish
log "Waiting for seed1 bm25only training to finish..."
while kill -0 2890806 2>/dev/null; do sleep 60; done
log "Seed1 bm25only training done."

# Eval seed1 bm25only on GPU 3
if [ -d "experiments/rankft_runA_bm25only_seed1/final" ]; then
    log "Evaluating seed1 bm25only..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only_seed1/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/rankft_runA_bm25only_seed1/eval_merged_rerank \
        --gpu_id 3 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/rankft_runA_bm25only_seed1_eval.log
    log "Seed1 bm25only eval done."
else
    log "ERROR: seed1 bm25only has no final/ directory"
fi

# If 1.5B model is available, run scale ablation
if [ -d "/data/chenlibin/models/Qwen2.5-1.5B-Instruct" ]; then
    log "Starting 1.5B scale ablation on GPU 3..."
    bash scripts/launch_scale_ablation.sh 1.5B 3 2>&1 | tee experiments/scale_1.5B.log
    log "1.5B scale ablation complete."
fi

# If 3B model is available, run scale ablation
if [ -d "/data/chenlibin/models/Qwen2.5-3B-Instruct" ]; then
    log "Starting 3B scale ablation on GPU 3..."
    bash scripts/launch_scale_ablation.sh 3B 3 2>&1 | tee experiments/scale_3B.log
    log "3B scale ablation complete."
fi

log "GPU3 pipeline complete."
