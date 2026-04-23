#!/bin/bash
set -euo pipefail
# Wait for hybrid-only v2 training to finish, then eval on GPU 2 (same GPU, now free)
# CF-Neg B deferred — decide based on CF-Neg A results first
cd /home/chenlibin/grepo_agent

echo "Waiting for hybrid-only v2 training to complete..."
while true; do
    if [ -d "experiments/rankft_hybrid_only_v2/final" ]; then
        echo "Hybrid-only v2 training done at $(date)"
        sleep 30  # let GPU memory free up

        # Launch hybrid-only eval on GPU 2 (now free)
        echo "Launching hybrid-only eval on GPU 2..."
        nohup bash scripts/eval_hybrid_only_reranker.sh 2 > logs/eval_hybrid_only_v2.log 2>&1 &
        echo "Hybrid eval PID=$!"
        echo "CF-Neg B deferred — check CF-Neg A results first"
        exit 0
    fi
    sleep 60
done
