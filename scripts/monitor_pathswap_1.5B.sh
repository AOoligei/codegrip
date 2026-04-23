#!/bin/bash
set -euo pipefail
# Wait for 7B PathSwap to finish on GPU 7, then run 1.5B PathSwap there
cd /home/chenlibin/grepo_agent

echo "Waiting for 7B PathSwap to finish on GPU 7..."
while true; do
    # Check if 7B hybrid (last eval for 7B) is done
    if [ -f "experiments/pathswap_eval/7B_hybrid/summary.json" ]; then
        echo "7B PathSwap fully done at $(date)"
        sleep 30
        echo "Launching 1.5B PathSwap eval on GPU 7..."
        bash scripts/eval_pathswap_1.5B.sh 7
        exit 0
    fi
    # Also check if 7B graph is done but hybrid hasn't started yet
    if [ -f "experiments/pathswap_eval/7B_graph/summary.json" ]; then
        echo "7B graph done, waiting for 7B hybrid..."
    fi
    sleep 120
done
