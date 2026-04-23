#!/bin/bash
# Quick status check: training progress + current results
cd /home/chenlibin/grepo_agent

echo "====================================="
echo "  CodeGRIP Status Report"
echo "  $(date)"
echo "====================================="

echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo "--- Training Progress ---"
for exp in exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
  log="experiments/${exp}.log"
  if [ -f "$log" ]; then
    current=$(grep -oP "\d+/\d+" "$log" | tail -1)
    loss=$(grep -oP "'loss': [\d.]+" "$log" | tail -1)
    total=$(echo "$current" | cut -d/ -f2)
    step=$(echo "$current" | cut -d/ -f1)
    pct=$(python3 -c "print(f'{100*$step/$total:.1f}')" 2>/dev/null)
    # Check if done
    if [ -d "experiments/$exp/stage2_sft/final" ]; then
      echo "  $exp: DONE"
    else
      echo "  $exp: $current ($pct%) $loss"
    fi
  fi
done

echo ""
echo "--- Best Results (vs GAT 14.80/31.51/37.40/41.25) ---"
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
printf "%-30s %7s %7s %7s %7s\n" "Method" "H@1" "H@5" "H@10" "H@20"
echo "--------------------------------------------------------------"

for exp in exp1_sft_only exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft; do
  for stage in eval_reranked eval_unified_expansion eval_filetree; do
    f="experiments/$exp/$stage/summary.json"
    if [ -f "$f" ]; then
      label="$exp ($stage)"
      $PYTHON -c "
import json
with open('$f') as fh:
    o = json.load(fh)['overall']
print(f'  {\"$label\":<28} {o.get(\"hit@1\",0):7.2f} {o.get(\"hit@5\",0):7.2f} {o.get(\"hit@10\",0):7.2f} {o.get(\"hit@20\",0):7.2f}')
" 2>/dev/null
      break  # Only show best available stage
    fi
  done
done
printf "  %-28s %7s %7s %7s %7s\n" "GAT (baseline)" "14.80" "31.51" "37.40" "41.25"

echo ""
echo "--- Disk ---"
df -h /home 2>/dev/null | tail -1
du -sh experiments/ 2>/dev/null
