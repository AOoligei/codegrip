#!/bin/bash
set -euo pipefail
# Monitor training progress for experiments
# Usage: bash scripts/monitor_training.sh
cd /home/chenlibin/grepo_agent

while true; do
    echo "=== $(date) ==="
    for exp in exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
        step=$(grep -oP '\d+/\d+ \[' experiments/${exp}.log 2>/dev/null | tail -1)
        if [ -n "$step" ]; then
            echo "  $exp: $step"
        else
            echo "  $exp: not started or finished"
        fi
    done
    echo
    sleep 300  # Check every 5 minutes
done
