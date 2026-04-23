#!/bin/bash
set -euo pipefail
# Monitor CF-Neg A training and auto-launch eval when done
cd /home/chenlibin/grepo_agent
while true; do
    if [ -d "experiments/rankft_cfneg_A/final" ]; then
        echo "CF-Neg A training done at $(date)"
        nohup bash scripts/eval_cfneg_A.sh 4 > logs/eval_cfneg_A.log 2>&1 &
        echo "Eval launched, PID=$!"
        exit 0
    fi
    if [ -f "experiments/rankft_cfneg_A/training_diagnostics.jsonl" ]; then
        STEP=$(tail -1 "experiments/rankft_cfneg_A/training_diagnostics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('step','?'))" 2>/dev/null || echo "?")
        echo "$(date +%H:%M): step $STEP/792"
    fi
    sleep 300
done
