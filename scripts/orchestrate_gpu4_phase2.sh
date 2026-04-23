#!/bin/bash
# GPU 4 phase 2: after gpu4 pipeline completes → BeetleBox Java eval → 14B scale ablation
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [GPU4-P2] $1" | tee -a orchestrate.log; }

# Wait for GPU4 phase 1 (PID 3563237) to finish
log "Waiting for GPU4 phase1 to finish..."
while kill -0 3563237 2>/dev/null; do sleep 60; done
log "GPU4 phase1 done."

# === BeetleBox Java eval ===
if [ -f "/data/chenlibin/beetlebox/java_test.jsonl" ]; then
    log "Starting BeetleBox Java eval on GPU 4..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runB_graph/best \
        --test_data /data/chenlibin/beetlebox/java_test.jsonl \
        --bm25_candidates /data/chenlibin/beetlebox/java_bm25_top500.jsonl \
        --output_dir experiments/beetlebox_java_eval \
        --gpu_id 4 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee experiments/beetlebox_java_eval.log
    log "BeetleBox Java eval done."

    # Print results
    $PYTHON -c "
import json
s = json.load(open('experiments/beetlebox_java_eval/summary.json'))['overall']
print(f'BeetleBox Java: R@1={s[\"recall@1\"]:.2f}% R@5={s[\"recall@5\"]:.2f}% R@10={s[\"recall@10\"]:.2f}%')
" 2>/dev/null || true
else
    log "WARN: BeetleBox java_test.jsonl not found, skipping"
fi

# === 14B scale ablation ===
if [ -d "/data/chenlibin/models/Qwen2.5-14B-Instruct" ]; then
    log "Starting 14B scale ablation on GPU 4..."
    bash scripts/launch_scale_ablation.sh 14B 4 2>&1 | tee experiments/scale_14B.log
    log "14B scale ablation complete."
else
    log "WARN: 14B model not found, skipping"
fi

log "GPU4 phase2 complete."
