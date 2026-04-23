#!/bin/bash
# Eval LoRA on SWE Lite normal + SHA-256 PathSwap.
# Usage: eval_2lora_pathswap_pair.sh <LORA_PATH> <OUTPUT_TAG> <GPU>
set -e
LORA=${1:?lora}
TAG=${2:?tag}
GPU=${3:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/eval_pathswap_bf16_${TAG}
mkdir -p $OUT

echo "=== $TAG NORMAL ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_bf16.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_final_top500.jsonl \
    --repo_dir data/swebench_lite/repos \
    --output_dir $OUT/normal \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size 8 2>&1 | tee logs/eval_pathswap_bf16_${TAG}_normal.log | tail -3

echo "=== $TAG PATHSWAP (SHA-256) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_bf16.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_lite_test_pathswap.jsonl \
    --bm25_candidates data/swebench_lite/swebench_bm25_pathswap.jsonl \
    --repo_dir data/swebench_lite/repos \
    --alias_map data/swebench_lite/pathswap_alias_map.json \
    --output_dir $OUT/pathswap \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size 8 2>&1 | tee logs/eval_pathswap_bf16_${TAG}_swap.log | tail -3

echo "=== $TAG SUMMARY ==="
for c in normal pathswap; do
  f=$OUT/$c/summary.json
  [ -f $f ] && $PY -c "import json; d=json.load(open('$f'))['overall']; print(f'$c: R@1={d[\"recall@1\"]:.2f} R@5={d[\"recall@5\"]:.2f}')"
done
