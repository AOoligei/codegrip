#!/bin/bash
set -euo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

echo "Monitoring PathSwap eval completion..."

while true; do
    # Check if parent pathswap process is still running
    if ! ps -p 842184 > /dev/null 2>&1; then
        echo "PathSwap eval script has finished!"
        break
    fi
    
    # Check for summary files
    for name in 0.5B 1.5B 3B 7B; do
        for pool in graph hybrid; do
            f="experiments/pathswap_eval/${name}_${pool}/summary.json"
            done_f="experiments/pathswap_eval/${name}_${pool}/.notified"
            if [ -f "$f" ] && [ ! -f "$done_f" ]; then
                echo "=== ${name}_${pool} COMPLETE ==="
                if cat "$f" | $PYTHON -c "import json,sys; d=json.load(sys.stdin)['overall']; print(f'  R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.4f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.4f}')"; then
                    touch "$done_f"
                fi
            fi
        done
    done
    sleep 30
done

echo ""
echo "=== Final PathSwap Results ==="
for name in 0.5B 1.5B 3B 7B; do
    for pool in graph hybrid; do
        f="experiments/pathswap_eval/${name}_${pool}/summary.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${name}_${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.4f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.4f}')" || echo "${name}_${pool}: RESULTS UNREADABLE"
        else
            echo "${name}_${pool}: NO RESULTS"
        fi
    done
done
