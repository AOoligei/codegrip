#!/bin/bash
# Evaluate all seed models on BM25-only pool for interaction analysis
# This gives us the 2x2 (pool × reranker) decomposition per seed
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
BM25_POOL=data/rankft/grepo_test_bm25_top500.jsonl
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
GPU=$1  # pass GPU as argument

echo "[$(date)] Starting interaction evals on GPU $GPU"

# Seeds and their checkpoint paths
declare -A GRAPH_CKPTS=(
    ["1"]="experiments/rankft_runB_graph_seed1/final"
    ["2"]="experiments/rankft_runB_graph_seed2/final"
    ["4"]="experiments/rankft_runB_graph_seed4/final"
)
declare -A BM25_CKPTS=(
    ["1"]="experiments/rankft_runA_bm25only_seed1/final"
    ["2"]="experiments/rankft_runA_bm25only_seed2/final"
    ["4"]="experiments/rankft_runA_bm25only_seed4/final"
)

# Determine which seeds this GPU handles
if [ "$GPU" = "5" ]; then
    SEEDS="1 2"
    TYPES="graph bm25"
elif [ "$GPU" = "6" ]; then
    SEEDS="4"
    TYPES="graph bm25"
else
    echo "Usage: $0 <5|6>"; exit 1
fi

for SEED in $SEEDS; do
    for TYPE in $TYPES; do
        if [ "$TYPE" = "graph" ]; then
            CKPT="${GRAPH_CKPTS[$SEED]}"
            OUTDIR="experiments/rankft_runB_graph_seed${SEED}/eval_bm25pool"
        else
            CKPT="${BM25_CKPTS[$SEED]}"
            OUTDIR="experiments/rankft_runA_bm25only_seed${SEED}/eval_bm25pool"
        fi

        if [ -f "$OUTDIR/summary.json" ]; then
            echo "[$(date)] SKIP: $OUTDIR already done"
            continue
        fi

        if [ ! -d "$CKPT" ]; then
            echo "[$(date)] SKIP: checkpoint $CKPT not found"
            continue
        fi

        echo "[$(date)] Evaluating seed$SEED $TYPE on BM25 pool..."
        rm -rf "$OUTDIR" 2>/dev/null
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
            --model_path $MODEL \
            --lora_path "$CKPT" \
            --test_data data/grepo_text/grepo_test.jsonl \
            --bm25_candidates "$BM25_POOL" \
            --output_dir "$OUTDIR" \
            --gpu_id 0 --top_k 200 --max_seq_length 512

        if [ -f "$OUTDIR/summary.json" ]; then
            R1=$(python3 -c "import json; print(f'{json.load(open(\"$OUTDIR/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
            echo "[$(date)] DONE: seed$SEED $TYPE on BM25 pool: R@1=$R1"
        fi
    done
done

echo "[$(date)] All interaction evals on GPU $GPU complete."
