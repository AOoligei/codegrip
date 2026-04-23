#!/bin/bash
# Evaluate a LoRA on SWE-bench Lite normal + shuffle_filenames perturbed.
# Usage: bash eval_gate2_both.sh <LORA_PATH> <OUTPUT_TAG> <GPU>
set -e
LORA=${1:?lora}
TAG=${2:?tag}
GPU=${3:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

OUT_DIR=/data/chenlibin/grepo_agent_experiments/eval_${TAG}
mkdir -p $OUT_DIR logs

echo "=== Eval $TAG on SWE Lite NORMAL ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_final_top500.jsonl \
    --repo_dir data/swebench_lite/repos \
    --output_dir $OUT_DIR/normal \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size 8 --include_code 2>&1 | tee logs/eval_${TAG}_normal.log | tail -5

echo "=== Eval $TAG on SWE Lite SHUFFLE_FILENAMES ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_perturb_shuffle_filenames_test.jsonl \
    --bm25_candidates data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl \
    --repo_dir data/swebench_lite/repos \
    --output_dir $OUT_DIR/shuffled \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size 8 --include_code 2>&1 | tee logs/eval_${TAG}_shuffled.log | tail -5

echo "=== Summary $TAG ==="
for cond in normal shuffled; do
  f=$OUT_DIR/$cond/summary.json
  [ -f "$f" ] && $PY -c "import json; d=json.load(open('$f'))['overall']; print(f'$cond: R@1={d[\"recall@1\"]:.2f} R@5={d[\"recall@5\"]:.2f}')"
done
