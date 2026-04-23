#!/bin/bash
set -euo pipefail
# Monitor CF-Neg training and auto-launch eval when done
VARIANT=$1  # A or B
GPU_TRAIN=$2
GPU_EVAL=$3
cd /home/chenlibin/grepo_agent

EXP_DIR="experiments/rankft_cfneg_${VARIANT}"
EVAL_SCRIPT="scripts/eval_cfneg_${VARIANT}.sh"

echo "Monitoring $EXP_DIR (train GPU=$GPU_TRAIN, eval GPU=$GPU_EVAL)"
echo "Start: $(date)"

while true; do
    if [ -d "${EXP_DIR}/final" ]; then
        echo "Training complete! Found ${EXP_DIR}/final at $(date)"
        echo "Launching eval on GPU $GPU_EVAL..."
        bash $EVAL_SCRIPT $GPU_EVAL
        echo "All done at $(date)"
        exit 0
    fi
    # Check if training process is still running
    if [ -f "${EXP_DIR}/training_diagnostics.jsonl" ]; then
        STEP=$(tail -1 "${EXP_DIR}/training_diagnostics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('step','?'))" 2>/dev/null || echo "?")
        echo "  $(date +%H:%M): step $STEP/792"
    fi
    sleep 300  # check every 5 min
done
