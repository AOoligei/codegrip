#!/bin/bash
# Wait for all 3 structural ablation evals to finish, then collect results
while true; do
  done=0
  for d in rankft_ablation_pathdist rankft_ablation_treeneighbor rankft_ablation_allstruct; do
    [ -f experiments/$d/eval_merged_rerank/summary.json ] && done=$((done+1))
  done
  echo "[$(date '+%H:%M')] $done/3 evals complete"
  [ $done -eq 3 ] && break
  sleep 300
done

echo "=== ALL EVALS COMPLETE ==="
cd /home/chenlibin/grepo_agent
python3 scripts/fill_structural_table.py
