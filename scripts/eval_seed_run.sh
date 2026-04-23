#!/bin/bash
# Usage: bash scripts/eval_seed_run.sh <exp_name> <gpu_id>
# Example: bash scripts/eval_seed_run.sh rankft_runB_graph_seed1 3
set -e
EXP=$1
GPU=$2
BASE=/home/chenlibin/grepo_agent

echo "[$(date)] Evaluating $EXP on GPU $GPU" | tee -a "$BASE/seed_robustness.log"

cd $BASE
conda run -n tgn python src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/$EXP/best \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/$EXP/eval_merged_rerank \
    --gpu_id $GPU \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee experiments/$EXP/eval.log

echo "[$(date)] Done evaluating $EXP" | tee -a "$BASE/seed_robustness.log"
