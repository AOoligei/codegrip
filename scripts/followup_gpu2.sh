#!/bin/bash
# Follow-up for GPU 2: run 7B scale ablation after orchestrate_all.sh completes
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [GPU2-FOLLOWUP] $1" | tee -a orchestrate.log; }

# Wait for GPU2 orchestrator (PID 3562329) to finish
log "Waiting for GPU2 orchestrator to finish..."
while kill -0 3562329 2>/dev/null; do sleep 120; done
log "GPU2 orchestrator done."

# Run 7B scale ablation (without SFT warmstart, for fair scale comparison)
if [ ! -d "experiments/scale_7B_graph/eval_merged_rerank" ]; then
    log "Starting 7B scale ablation on GPU 2..."
    bash scripts/launch_scale_ablation.sh 7B 2 2>&1 | tee experiments/scale_7B.log
    log "7B scale ablation complete."
else
    log "7B scale ablation already done, skipping."
fi

log "GPU2 follow-up complete."
