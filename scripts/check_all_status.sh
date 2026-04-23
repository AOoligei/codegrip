#!/bin/bash
cd /home/chenlibin/grepo_agent
echo "=== $(date) ==="
echo ""

# Training progress
echo "--- Training Progress (total ~986 steps) ---"
for d in scale_3B_graph scale_7B_bm25only rankft_runA_bm25only_seed3 rankft_runA_bm25only_seed4; do
    if [ -d "experiments/$d/final" ]; then
        echo "$d: COMPLETE"
    elif [ -f "experiments/$d/training_diagnostics.jsonl" ]; then
        step=$(tail -1 "experiments/$d/training_diagnostics.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['step'])" 2>/dev/null || echo "?")
        pct=$(python3 -c "print(f'{int($step)/986*100:.1f}%')" 2>/dev/null || echo "?")
        echo "$d: step $step ($pct)"
    else
        echo "$d: not started"
    fi
done
echo ""

# Eval results
echo "--- Eval Results ---"
for d in scale_7B_graph scale_7B_bm25only scale_3B_graph scale_3B_bm25only \
         rankft_runA_bm25only_seed3 rankft_runA_bm25only_seed4 \
         beetlebox_java_eval beetlebox_java_bm25only; do
    eval_dir="experiments/$d/eval_merged_rerank"
    [ "$d" = "beetlebox_java_eval" ] && eval_dir="experiments/$d"
    [ "$d" = "beetlebox_java_bm25only" ] && eval_dir="experiments/$d"

    if [ -f "$eval_dir/summary.json" ]; then
        r1=$(python3 -c "import json; d=json.load(open('$eval_dir/summary.json'))['overall']; print(f'{d[\"recall@1\"]:.2f}')" 2>/dev/null || echo "?")
        echo "$d: R@1=$r1"
    elif [ -d "$eval_dir" ] && ls "$eval_dir"/*.json 2>/dev/null | head -1 | grep -q .; then
        echo "$d: eval running (partial results)"
    else
        echo "$d: pending"
    fi
done
echo ""

# GPU allocation
echo "--- GPU Processes ---"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader 2>/dev/null | while read line; do
    pid=$(echo "$line" | awk -F', ' '{print $2}')
    cmd=$(ps -p $pid -o args= 2>/dev/null | grep -oP '(?:--output_dir |--lora_path )\K\S+' | head -1)
    echo "$line  [$cmd]"
done
