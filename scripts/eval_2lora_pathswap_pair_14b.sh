#!/bin/bash
# Eval 14B LoRA on SWE-bench Lite normal + SHA-256 PathSwap.
# Fork of eval_2lora_pathswap_pair.sh with 14B base model + smaller batch.
# Usage: eval_2lora_pathswap_pair_14b.sh <LORA_PATH> <OUTPUT_TAG> <GPU>
set -euo pipefail
LORA=${1:?lora}
TAG=${2:?tag}
GPU=${3:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=${MODEL_PATH:-/data/chenlibin/models/Qwen2.5-14B-Instruct}
BATCH=${SCORE_BATCH_SIZE:-4}
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/eval_pathswap_${TAG}
mkdir -p "$OUT" logs
# Remove stale summary so failures cannot masquerade as success
rm -f "$OUT/normal/summary.json" "$OUT/pathswap/summary.json"

# Sanity preflight
test -d "$MODEL" || { echo "ERROR: missing model $MODEL" >&2; exit 2; }
test -d "$LORA" || { echo "ERROR: missing lora $LORA" >&2; exit 2; }
test -d data/swebench_lite/repos || { echo "ERROR: missing repo_dir data/swebench_lite/repos" >&2; exit 2; }
test -f data/swebench_lite/swebench_lite_test.jsonl || { echo "ERROR: missing lite test" >&2; exit 2; }
test -f data/rankft/swebench_bm25_final_top500.jsonl || { echo "ERROR: missing lite bm25" >&2; exit 2; }
test -f data/swebench_lite/swebench_lite_test_pathswap.jsonl || { echo "ERROR: missing lite pathswap test" >&2; exit 2; }
test -f data/swebench_lite/swebench_bm25_pathswap.jsonl || { echo "ERROR: missing lite pathswap bm25" >&2; exit 2; }
test -f data/swebench_lite/pathswap_alias_map.json || { echo "ERROR: missing lite alias map" >&2; exit 2; }

echo "=== $TAG NORMAL (14B) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path "$MODEL" \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_final_top500.jsonl \
    --repo_dir data/swebench_lite/repos \
    --output_dir "$OUT/normal" \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size "$BATCH" 2>&1 | tee logs/eval_pathswap_${TAG}_normal.log | tail -n 5

echo "=== $TAG PATHSWAP (SHA-256, 14B) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path "$MODEL" \
    --lora_path "$LORA" \
    --test_data data/swebench_lite/swebench_lite_test_pathswap.jsonl \
    --bm25_candidates data/swebench_lite/swebench_bm25_pathswap.jsonl \
    --repo_dir data/swebench_lite/repos \
    --alias_map data/swebench_lite/pathswap_alias_map.json \
    --output_dir "$OUT/pathswap" \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size "$BATCH" 2>&1 | tee logs/eval_pathswap_${TAG}_swap.log | tail -n 5

echo "=== $TAG SUMMARY (14B) ==="
for c in normal pathswap; do
  f=$OUT/$c/summary.json
  [ -f "$f" ] && $PY -c "import json; d=json.load(open('$f'))['overall']; print(f'$c: R@1={d[\"recall@1\"]:.2f} R@3={d[\"recall@3\"]:.2f} R@5={d[\"recall@5\"]:.2f} R@10={d[\"recall@10\"]:.2f} R@20={d[\"recall@20\"]:.2f}')"
done
