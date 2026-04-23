#!/bin/bash
# GPU 4 pipeline: after seed1 graph eval → 0.5B scale ablation → SWE-bench fair eval backup
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [GPU4] $1" | tee -a orchestrate.log; }

# Wait for seed1 graph eval (PID 3493725) to finish
log "Waiting for seed1 graph eval to finish..."
while kill -0 3493725 2>/dev/null; do sleep 30; done
log "Seed1 graph eval done."

# Check if 0.5B model exists
if [ ! -d "/data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct" ]; then
    log "ERROR: 0.5B model not found at /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct"
    exit 1
fi

# Run 0.5B scale ablation on GPU 4
log "Starting 0.5B scale ablation..."
bash scripts/launch_scale_ablation.sh 0.5B 4 2>&1 | tee experiments/scale_0.5B.log

log "0.5B scale ablation complete."

# Eval seed1 bm25only if training is done by now
if [ -d "experiments/rankft_runA_bm25only_seed1/final" ]; then
    log "Evaluating seed1 bm25only..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only_seed1/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/rankft_runA_bm25only_seed1/eval_merged_rerank \
        --gpu_id 4 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/rankft_runA_bm25only_seed1_eval.log
    log "Seed1 bm25only eval done."
fi

# Eval seed2 bm25only if training is done by now
if [ -d "experiments/rankft_runA_bm25only_seed2/final" ]; then
    log "Evaluating seed2 bm25only..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only_seed2/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/rankft_runA_bm25only_seed2/eval_merged_rerank \
        --gpu_id 4 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/rankft_runA_bm25only_seed2_eval.log
    log "Seed2 bm25only eval done."
fi

log "GPU4 pipeline complete."
