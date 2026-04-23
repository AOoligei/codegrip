#!/bin/bash
cd /home/chenlibin/grepo_agent

# Clean code_residual_7b checkpoints (keep last 3)
for DIR in experiments/code_residual_7b experiments/*/; do
  checkpoints=($(ls -d ${DIR}checkpoint-* 2>/dev/null | sort -t'-' -k2 -n))
  n=${#checkpoints[@]}
  if [ $n -gt 3 ]; then
    for ((i=0; i<n-3; i++)); do
      rm -rf "${checkpoints[$i]}"
    done
    echo "$(date): $DIR deleted $((n-3)) checkpoints"
  fi
done

# Clean /tmp claude files older than 1 hour
find /tmp -maxdepth 2 -name "claude-*" -mmin +60 -exec rm -rf {} + 2>/dev/null

# Alert if root disk < 10G free
free_kb=$(df / | tail -1 | awk '{print $4}')
if [ "$free_kb" -lt 10485760 ]; then
  echo "$(date): WARNING root disk low: ${free_kb}KB free" >> /home/chenlibin/grepo_agent/logs/disk_alert.log
fi
