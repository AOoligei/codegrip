#!/bin/bash
set -euo pipefail
# Monitor all running experiments and auto-collect results
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent

check_interaction() {
    # Check if we have enough bm25pool results for interaction analysis
    local count=0
    for seed in 42 1 2 3 4; do
        if [ "$seed" = "42" ]; then
            g_dir="experiments/rankft_runB_graph/eval_bm25pool"
            b_dir="experiments/rankft_runA_bm25only/eval_bm25pool"
            g_exp="experiments/rankft_runB_graph/eval_merged_rerank"
            b_exp="experiments/rankft_runA_bm25only/eval_merged_rerank"
        else
            g_dir="experiments/rankft_runB_graph_seed${seed}/eval_bm25pool"
            b_dir="experiments/rankft_runA_bm25only_seed${seed}/eval_bm25pool"
            g_exp="experiments/rankft_runB_graph_seed${seed}/eval_merged_rerank"
            b_exp="experiments/rankft_runA_bm25only_seed${seed}/eval_merged_rerank"
        fi
        # Need all 4: graph+bm25 on both pools
        if [ -f "$g_dir/summary.json" ] && [ -f "$b_dir/summary.json" ] && \
           [ -f "$g_exp/summary.json" ] && [ -f "$b_exp/summary.json" ]; then
            count=$((count + 1))
        fi
    done
    # seed42 graph on bm25pool may use old eval_bm25_only_k200
    if [ -f "experiments/rankft_runB_graph/eval_bm25_only_k200/summary.json" ] && \
       [ -f "experiments/rankft_runA_bm25only/eval_bm25pool/summary.json" ] && \
       [ -f "experiments/rankft_runB_graph/eval_merged_rerank/summary.json" ] && \
       [ -f "experiments/rankft_runA_bm25only/eval_merged_rerank/summary.json" ]; then
        # seed42 complete via fallback
        count=$((count + 1))
    fi
    echo $count
}

echo "[$(date)] Starting monitor..."
INTERACTION_RAN=false

while true; do
    COMPLETE=$(check_interaction)
    echo "[$(date)] Interaction seeds complete: $COMPLETE/5"

    if [ "$COMPLETE" -ge 2 ] && [ "$INTERACTION_RAN" = "false" ]; then
        echo "[$(date)] Running interaction analysis ($COMPLETE seeds)..."
        $PYTHON scripts/analyze_interaction.py 2>&1
        INTERACTION_RAN=true
    fi

    # Re-run analysis if more seeds complete
    NEW_COMPLETE=$(check_interaction)
    if [ "$NEW_COMPLETE" -gt "$COMPLETE" ] && [ "$INTERACTION_RAN" = "true" ]; then
        echo "[$(date)] More seeds complete ($NEW_COMPLETE), re-running analysis..."
        $PYTHON scripts/analyze_interaction.py 2>&1
    fi

    # Check if seed3 bm25 eval just finished
    if [ -f "experiments/rankft_runA_bm25only_seed3/eval_merged_rerank/summary.json" ]; then
        R1=$($PYTHON -c "import json; print(f'{json.load(open(\"experiments/rankft_runA_bm25only_seed3/eval_merged_rerank/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
        echo "[$(date)] seed3 bm25 expanded: R@1=$R1"
    fi

    # Check 3B bm25
    if [ -f "experiments/scale_3B_bm25only/eval_merged_rerank/summary.json" ]; then
        R1=$($PYTHON -c "import json; print(f'{json.load(open(\"experiments/scale_3B_bm25only/eval_merged_rerank/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
        echo "[$(date)] 3B bm25 expanded: R@1=$R1"
    fi

    # Check if all processes are done
    RUNNING=$(pgrep -u chenlibin -fc "train_rankft\|eval_rankft" 2>/dev/null || echo 0)
    if [ "$RUNNING" = "0" ]; then
        echo "[$(date)] All processes done!"
        echo "[$(date)] Final interaction analysis:"
        $PYTHON scripts/analyze_interaction.py 2>&1
        break
    fi

    sleep 300  # Check every 5 minutes
done
