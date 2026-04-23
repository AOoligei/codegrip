#!/bin/bash
# v2: Kill high-CPU graftcp + unfreeze any ptrace-stopped training/eval processes
cd /home/chenlibin/grepo_agent
while true; do
    # Kill any high-CPU graftcp processes (the ones actively ptracing)
    for pid in $(pgrep -u chenlibin -f "graftcp -a" 2>/dev/null); do
        cpu=$(ps -p $pid -o pcpu= 2>/dev/null | xargs)
        if python3 -c "import sys; sys.exit(0 if float('${cpu:-0}') > 30 else 1)" 2>/dev/null; then
            kill $pid 2>/dev/null && echo "[$(date)] Killed graftcp $pid (${cpu}% CPU)"
        fi
    done
    # Unfreeze any traced training/eval processes
    for pid in $(pgrep -u chenlibin -f "train_rankft\|eval_rankft" 2>/dev/null); do
        state=$(cat /proc/$pid/stat 2>/dev/null | awk '{print $3}')
        if [ "$state" = "t" ] || [ "$state" = "T" ]; then
            kill -STOP $pid 2>/dev/null; sleep 0.5; kill -CONT $pid 2>/dev/null
            echo "[$(date)] Unfroze PID $pid"
        fi
    done
    sleep 10
done
