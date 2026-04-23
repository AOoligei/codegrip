#!/bin/bash
set -euo pipefail
# Monitor all experiments, log to experiments/monitor.log
# Usage: nohup bash scripts/monitor_all.sh &
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd "$BASE"

LOG=experiments/monitor.log

while true; do
    echo "" >> $LOG
    echo "=== $(date) ===" >> $LOG

    # Training progress
    for f in experiments/scale_*_graph/training_diagnostics.jsonl \
             experiments/rankft_runB_graph_seed*/training_diagnostics.jsonl \
             experiments/rankft_runA_bm25only_seed*/training_diagnostics.jsonl \
             experiments/scale_*_bm25only/training_diagnostics.jsonl; do
        if [ -f "$f" ]; then
            name=$(echo "$f" | sed 's|experiments/||;s|/training_diagnostics.jsonl||')
            step=$(tail -1 "$f" 2>/dev/null | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d['step'])" 2>/dev/null || echo "?")
            echo "  $name: step $step/792" >> $LOG
        fi
    done

    # Completed evals
    for f in experiments/*/eval_merged_rerank/summary.json \
             experiments/beetlebox_*/summary.json \
             experiments/path_anon_*/eval/summary.json; do
        if [ -f "$f" ]; then
            name=$(echo "$f" | sed 's|experiments/||;s|/eval.*/summary.json||;s|/summary.json||')
            r1=$(python3 -c "import json;d=json.load(open('$f'))['overall'];print(f'{d.get(\"recall@1\",d.get(\"hit@1\",0)):.2f}')" 2>/dev/null || echo "N/A")
            echo "  [DONE] $name: R@1=$r1" >> $LOG
        fi
    done

    # Orchestrator PIDs
    for pid in 3562328 3566837 3563237 4126762 143142; do
        status=$(kill -0 $pid 2>/dev/null && echo "RUNNING" || echo "DONE")
        echo "  orch $pid: $status" >> $LOG
    done

    # Check if all orchestrators are done
    all_done=true
    for pid in 3562328 3566837 3563237 4126762 143142; do
        kill -0 $pid 2>/dev/null && all_done=false
    done

    if $all_done; then
        echo "=== ALL ORCHESTRATORS COMPLETE ===" >> $LOG
        echo "Running final collection..." >> $LOG
        $PYTHON scripts/collect_seed_results.py >> $LOG 2>&1
        $PYTHON scripts/collect_scale_results.py >> $LOG 2>&1
        $PYTHON scripts/fill_paper_results.py >> $LOG 2>&1
        echo "=== MONITOR COMPLETE ===" >> $LOG
        break
    fi

    sleep 1800  # 30 min
done
