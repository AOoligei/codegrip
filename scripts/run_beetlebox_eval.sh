#!/bin/bash
# Cross-language evaluation on BeetleBox Java (740 examples)
# Tests: Python-trained reranker on Java bugs
# Runs: baseline + shuffle_filenames perturbation
set -euo pipefail

GPU=${1:-4}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/data/chenlibin/grepo_agent_experiments/beetlebox_java
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=/home/chenlibin/grepo_agent/experiments/rankft_runB_graph/best

mkdir -p $OUT

echo "=== BeetleBox Java Cross-Language Eval ==="
echo "GPU: $GPU, Output: $OUT"

# Step 1: Generate perturbation data
echo "--- Generating shuffle_filenames perturbation ---"
$PYTHON scripts/perturb_beetlebox.py

# Step 2: Baseline eval (no perturbation)
echo "--- Baseline eval ---"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_rankft_4bit.py \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data /data/chenlibin/beetlebox/java_test.jsonl \
    --bm25_candidates /data/chenlibin/beetlebox/java_bm25_top500.jsonl \
    --output_dir $OUT/eval_baseline \
    --gpu_id 0 \
    --top_k 200 \
    2>&1 | tee $OUT/log_baseline.txt

# Step 3: Shuffle filenames eval
echo "--- Shuffle filenames eval ---"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_rankft_4bit.py \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data /data/chenlibin/beetlebox/perturb_shuffle_filenames/java_test.jsonl \
    --bm25_candidates /data/chenlibin/beetlebox/perturb_shuffle_filenames/java_bm25_top500.jsonl \
    --output_dir $OUT/eval_shuffle_filenames \
    --gpu_id 0 \
    --top_k 200 \
    2>&1 | tee $OUT/log_shuffle_filenames.txt

echo "=== DONE ==="
# Print results
for d in eval_baseline eval_shuffle_filenames; do
    if [ -f "$OUT/$d/summary.json" ]; then
        echo "$d: $($PYTHON -c "import json; d=json.load(open('$OUT/$d/summary.json')); print(f'R@1={d[\"overall\"][\"hit@1\"]:.2f}%')")"
    fi
done
