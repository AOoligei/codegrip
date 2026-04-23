#!/bin/bash
# Watchdog: kill recover_seeds_bm25.sh after seed3 eval completes
# This prevents it from rm -rf'ing seed4 which is already training on GPU 5
set -e
RECOVER_PID=1812710
SEED3_EVAL="experiments/rankft_runA_bm25only_seed3/eval_merged_rerank/summary.json"

echo "[$(date)] Watchdog started. Waiting for seed3 eval to complete..."

while true; do
    # Check if recover script is still running
    if ! kill -0 $RECOVER_PID 2>/dev/null; then
        echo "[$(date)] recover_seeds_bm25.sh already exited. Watchdog done."
        exit 0
    fi

    # Check if seed3 eval is done
    if [ -f "$SEED3_EVAL" ]; then
        echo "[$(date)] Seed3 eval complete. Killing recover_seeds_bm25.sh to protect seed4..."
        kill $RECOVER_PID 2>/dev/null
        sleep 2
        # Force kill if still alive
        kill -9 $RECOVER_PID 2>/dev/null || true
        echo "[$(date)] recover_seeds_bm25.sh killed. Seed4 on GPU 5 is safe."

        # Print seed3 result
        python3 -c "
import json
d = json.load(open('$SEED3_EVAL'))['overall']
print(f'Seed3 BM25 R@1={d[\"recall@1\"]:.2f}')
"
        exit 0
    fi

    sleep 30
done
