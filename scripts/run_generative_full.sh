#!/bin/bash
# Run generative listwise reranker on FULL test set (1704 examples)
# Fixes: previous run only used 200 examples
# Uses top-20 candidates (not top-10) for fairer comparison
set -euo pipefail

GPU=${1:-4}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/data/chenlibin/grepo_agent_experiments/generative_reranker_full

mkdir -p $OUT

echo "=== Generative Reranker Full Eval ==="
echo "GPU: $GPU, Output: $OUT"

for COND in none shuffle_filenames shuffle_dirs; do
    echo "--- $COND ---"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_generative_reranker.py \
        --gpu_id 0 \
        --perturb $COND \
        --max_examples 99999 \
        --top_k 20 \
        2>&1 | tee $OUT/log_${COND}.txt

    # Copy results
    cp experiments/generative_reranker/results_${COND}.json $OUT/results_${COND}.json 2>/dev/null || true
done

echo "=== ALL DONE ==="
