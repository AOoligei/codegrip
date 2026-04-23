#!/bin/bash
set -euo pipefail
# Monitor SWE-bench reranker training (GPU 1), launch eval when done
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

echo "=== SWE-bench monitor started at $(date) ==="

# Wait for training to finish
while [ ! -f experiments/swebench_rankft_bm25/final/adapter_model.safetensors ]; do
    sleep 300
done
echo "=== SWE-bench training done at $(date) ==="

# Launch evals on GPU 3
bash scripts/swebench_eval_oracle_fallacy.sh 3

echo "=== All SWE-bench evals done at $(date) ==="
