#!/bin/bash
# Auto-SIGCONT any training/eval process that gets frozen by graftcp ptrace
cd /home/chenlibin/grepo_agent
PIDS="2806904 2811825 2375086 2173701 2019885 2806902"

while true; do
    for pid in $PIDS; do
        if kill -0 $pid 2>/dev/null; then
            state=$(cat /proc/$pid/stat 2>/dev/null | awk '{print $3}')
            if [ "$state" = "t" ] || [ "$state" = "T" ]; then
                kill -STOP $pid 2>/dev/null
                sleep 0.5
                kill -CONT $pid 2>/dev/null
                echo "[$(date)] Unfroze PID $pid (was in state $state)"
            fi
        fi
    done
    sleep 15
done
