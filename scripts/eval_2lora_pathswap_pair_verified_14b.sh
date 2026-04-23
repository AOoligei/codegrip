#!/bin/bash
# Eval 14B LoRA on SWE-bench Verified normal + SHA-256 PathSwap.
# Fork of eval_2lora_pathswap_pair_verified.sh with 14B base model + smaller batch.
# Usage: eval_2lora_pathswap_pair_verified_14b.sh <LORA_PATH> <OUTPUT_TAG> <GPU>
set -euo pipefail
LORA=${1:?lora}
TAG=${2:?tag}
GPU=${3:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=${MODEL_PATH:-/data/chenlibin/models/Qwen2.5-14B-Instruct}
BATCH=${SCORE_BATCH_SIZE:-4}
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/eval_verified_pathswap_${TAG}
mkdir -p "$OUT" logs
# Remove stale summary so failures cannot masquerade as success
rm -f "$OUT/normal/summary.json" "$OUT/pathswap/summary.json"

VERIFIED_DIR=/data/chenlibin/grepo_agent_experiments/swebench_verified
TEST_NORMAL=$VERIFIED_DIR/swebench_verified_prepared.jsonl
BM25_NORMAL=$VERIFIED_DIR/swebench_verified_bm25_top500.jsonl
TEST_SWAP=$VERIFIED_DIR/swebench_verified_test_pathswap.jsonl
BM25_SWAP=$VERIFIED_DIR/swebench_verified_bm25_pathswap.jsonl
ALIAS_MAP=$VERIFIED_DIR/pathswap_alias_map_verified.json
REPOS=data/swebench_lite/repos   # Verified shares same 12 repos as Lite

# Sanity preflight
test -d "$MODEL" || { echo "ERROR: missing model $MODEL" >&2; exit 2; }
test -d "$LORA" || { echo "ERROR: missing lora $LORA" >&2; exit 2; }
test -d "$REPOS" || { echo "ERROR: missing repo_dir $REPOS" >&2; exit 2; }
test -f "$TEST_NORMAL" || { echo "ERROR: missing $TEST_NORMAL" >&2; exit 2; }
test -f "$BM25_NORMAL" || { echo "ERROR: missing $BM25_NORMAL" >&2; exit 2; }
test -f "$TEST_SWAP" || { echo "ERROR: missing $TEST_SWAP" >&2; exit 2; }
test -f "$BM25_SWAP" || { echo "ERROR: missing $BM25_SWAP" >&2; exit 2; }
test -f "$ALIAS_MAP" || { echo "ERROR: missing $ALIAS_MAP" >&2; exit 2; }

echo "=== $TAG VERIFIED NORMAL (14B) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path "$MODEL" \
    --lora_path "$LORA" \
    --test_data "$TEST_NORMAL" \
    --bm25_candidates "$BM25_NORMAL" \
    --repo_dir "$REPOS" \
    --output_dir "$OUT/normal" \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size "$BATCH" \
    2>&1 | tee logs/eval_verified_pathswap_${TAG}_normal.log | tail -n 5

echo "=== $TAG VERIFIED PATHSWAP (SHA-256, 14B) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_codeaware_4bit.py \
    --model_path "$MODEL" \
    --lora_path "$LORA" \
    --test_data "$TEST_SWAP" \
    --bm25_candidates "$BM25_SWAP" \
    --repo_dir "$REPOS" \
    --alias_map "$ALIAS_MAP" \
    --output_dir "$OUT/pathswap" \
    --include_code --code_max_lines 50 \
    --gpu_id 0 --top_k 100 --max_seq_length 768 --score_batch_size "$BATCH" \
    2>&1 | tee logs/eval_verified_pathswap_${TAG}_swap.log | tail -n 5

echo "=== $TAG VERIFIED SUMMARY (14B) ==="
for c in normal pathswap; do
  f=$OUT/$c/summary.json
  [ -f "$f" ] && $PY -c "import json; d=json.load(open('$f'))['overall']; print(f'$c: R@1={d[\"recall@1\"]:.2f} R@3={d[\"recall@3\"]:.2f} R@5={d[\"recall@5\"]:.2f} R@10={d[\"recall@10\"]:.2f} R@20={d[\"recall@20\"]:.2f}')"
done
