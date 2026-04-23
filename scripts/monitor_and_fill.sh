#!/bin/bash
# Monitor running evals and fill results when done
# Usage: nohup bash scripts/monitor_and_fill.sh >> monitor.log 2>&1 &
set -euo pipefail
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd "$BASE"

echo "[$(date)] Monitor started"

# Wait for samedir eval
while [ ! -f experiments/rankft_ablation_samedir/eval_merged_rerank/summary.json ]; do
    sleep 60
done
echo "[$(date)] samedir eval DONE"
$PYTHON scripts/fill_structural_table.py

# Wait for seed1 graph eval
while [ ! -f experiments/rankft_runB_graph_seed1/eval_merged_rerank/summary.json ]; do
    sleep 60
done
echo "[$(date)] seed1 graph eval DONE"

# Try collecting whatever seed results are available
$PYTHON scripts/collect_seed_results.py

echo "[$(date)] Monitor finished"
