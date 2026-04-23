#!/bin/bash
# Evaluate SPHT LoRA on clean + PathSwap + Code-Crucial + SWE-bench
set -e
GPU=${1:-4}
LORA=${2:-experiments/rankft_spht/best}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/spht_eval
mkdir -p $OUT

# Clean GREPO (graph pool)
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path $LORA \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir $OUT/clean_grepo \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

# PathSwap GREPO
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path $LORA \
    --test_data data/pathswap/grepo_test_pathswap.jsonl \
    --bm25_candidates data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl \
    --output_dir $OUT/pathswap_grepo \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "=== SPHT Eval Summary ==="
for e in clean_grepo pathswap_grepo; do
    f=$OUT/$e/summary.json
    [ -f "$f" ] && $PY -c "import json; d=json.load(open('$f'))['overall']; print(f'  $e R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}%')"
done
