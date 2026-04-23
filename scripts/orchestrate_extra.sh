#!/bin/bash
# Extra experiments: seeds 3,4 on GPU 1 (after seed2 bm25) + path anonymization
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [EXTRA] $1" | tee -a orchestrate.log; }

# === Phase 1: Wait for seed2 bm25 training (GPU 1, PID 3501294) ===
log "Waiting for seed2 bm25 training to finish..."
while kill -0 3501294 2>/dev/null; do sleep 60; done
log "Seed2 bm25 training done."

# Eval seed2 bm25 on GPU 1
if [ -d "experiments/rankft_runA_bm25only_seed2/final" ]; then
    log "Evaluating seed2 bm25 on GPU 1..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only_seed2/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/rankft_runA_bm25only_seed2/eval_merged_rerank \
        --gpu_id 1 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/rankft_runA_bm25only_seed2_eval.log
    log "Seed2 bm25 eval done."
fi

# === Phase 2: Run seeds 3,4 on GPU 1 ===
log "Starting extra seeds (3,4) on GPU 1..."
bash scripts/run_extra_seeds.sh 1 2>&1 | tee experiments/extra_seeds.log
log "Extra seeds complete."

# === Phase 3: Path anonymization on GPU 1 ===
log "Starting path anonymization control on GPU 1..."
$PYTHON scripts/path_anonymization_eval.py --gpu_id 1 --modes dir full \
    2>&1 | tee experiments/path_anon.log
log "Path anonymization complete."

# === Phase 4: Re-eval seed42 bm25 on merged candidates (for fair comparison) ===
if [ ! -f "experiments/rankft_runA_bm25only/eval_merged_rerank/summary.json" ]; then
    log "Re-evaluating seed42 bm25 on merged candidates..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only/best \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir experiments/rankft_runA_bm25only/eval_merged_rerank \
        --gpu_id 1 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/rankft_runA_bm25only_merged_eval.log
    log "Seed42 bm25 merged eval done."
fi

# Final collection
log "Collecting all seed results..."
$PYTHON scripts/collect_seed_results.py
$PYTHON scripts/fill_paper_results.py

log "=== EXTRA PIPELINE COMPLETE ==="
