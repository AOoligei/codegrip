#!/bin/bash
# v3: Only unfreeze ptrace-stopped processes, do NOT kill graftcp
cd /home/chenlibin/grepo_agent
while true; do
    for pid in $(pgrep -u chenlibin -f "train_rankft\|eval_rankft" 2>/dev/null); do
        state=$(cat /proc/$pid/stat 2>/dev/null | awk '{print $3}')
        if [ "$state" = "t" ] || [ "$state" = "T" ]; then
            kill -STOP $pid 2>/dev/null; sleep 0.5; kill -CONT $pid 2>/dev/null
            echo "[$(date)] Unfroze PID $pid"
        fi
    done
    sleep 10
done
