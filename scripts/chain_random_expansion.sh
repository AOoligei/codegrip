#!/bin/bash
set -eo pipefail
cd /home/chenlibin/grepo_agent

# Wait for seed42 graph bm25pool eval (GPU0, PID 287413) to finish
echo "[$(date)] Waiting for seed42 graph bm25pool eval (PID 287413) to finish..."
while kill -0 287413 2>/dev/null; do sleep 60; done
echo "[$(date)] GPU0 free. Starting random expansion eval."

# Run random expansion on GPU0
bash scripts/eval_random_expansion.sh 0

echo "[$(date)] Random expansion control complete!"
