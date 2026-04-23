#!/bin/bash
# Fair SWE-bench Lite comparison: graph-hard vs bm25-only
# Same BM25 candidates, same top_k=50, same max_seq_length=1024
# Usage: bash scripts/swebench_fair_comparison.sh <gpu_id>
set -e

GPU=$1
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

echo "[$(date)] Starting fair SWE-bench comparison on GPU $GPU"

# Graph-hard model
echo "[$(date)] Evaluating graph-hard on SWE-bench Lite..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir experiments/rankft_runB_graph/eval_swebench_fair \
    --gpu_id $GPU \
    --top_k 50 \
    --max_seq_length 1024 \
    2>&1 | tee experiments/rankft_runB_graph/eval_swebench_fair.log

echo "[$(date)] Graph-hard done."

# BM25-only model
echo "[$(date)] Evaluating bm25-only on SWE-bench Lite..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runA_bm25only/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir experiments/rankft_runA_bm25only/eval_swebench_fair \
    --gpu_id $GPU \
    --top_k 50 \
    --max_seq_length 1024 \
    2>&1 | tee experiments/rankft_runA_bm25only/eval_swebench_fair.log

echo "[$(date)] BM25-only done."

# Compare
echo ""
echo "=== Fair SWE-bench Lite Comparison ==="
$PYTHON -c "
import json
g = json.load(open('experiments/rankft_runB_graph/eval_swebench_fair/summary.json'))['overall']
b = json.load(open('experiments/rankft_runA_bm25only/eval_swebench_fair/summary.json'))['overall']
print(f'Graph-hard: H@1={g[\"hit@1\"]:.1f}% H@5={g[\"hit@5\"]:.1f}% H@10={g[\"hit@10\"]:.1f}%')
print(f'BM25-only:  H@1={b[\"hit@1\"]:.1f}% H@5={b[\"hit@5\"]:.1f}% H@10={b[\"hit@10\"]:.1f}%')
print(f'Delta H@1:  {g[\"hit@1\"]-b[\"hit@1\"]:+.1f}%')
"

echo "[$(date)] Fair comparison complete."
