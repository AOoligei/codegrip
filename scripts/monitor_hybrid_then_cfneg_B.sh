#!/bin/bash
set -euo pipefail
# Wait for hybrid-only to finish, launch its eval, then launch CF-Neg B training on freed GPU
EVAL_GPU=4  # eval hybrid-only on GPU 4
cd /home/chenlibin/grepo_agent

echo "Waiting for hybrid-only training to complete..."
while true; do
    if [ -d "experiments/rankft_hybrid_only_v2/final" ]; then
        echo "Hybrid-only training done at $(date)"

        # Launch hybrid-only eval on a different GPU
        echo "Launching hybrid-only eval on GPU $EVAL_GPU..."
        nohup bash scripts/eval_hybrid_only_reranker.sh $EVAL_GPU > logs/eval_hybrid_only_v2.log 2>&1 &
        echo "Hybrid eval PID=$!"

        # Wait a moment for GPU 2 memory to be freed
        sleep 30

        # Launch CF-Neg B on GPU 2 (now free)
        echo "Launching CF-Neg B training on GPU 2..."
        nohup bash scripts/train_cfneg_B.sh 2 > logs/train_cfneg_B.log 2>&1 &
        CFNEG_B_PID=$!
        echo "CF-Neg B PID=$CFNEG_B_PID"

        # Now monitor CF-Neg B and auto-eval when done
        echo "Monitoring CF-Neg B..."
        while true; do
            if [ -d "experiments/rankft_cfneg_B/final" ]; then
                echo "CF-Neg B training done at $(date)"
                nohup bash scripts/eval_cfneg_B.sh 5 > logs/eval_cfneg_B.log 2>&1 &
                echo "CF-Neg B eval launched"
                exit 0
            fi
            sleep 300
        done
    fi
    sleep 120
done
