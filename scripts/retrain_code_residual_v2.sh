#!/bin/bash
# Retrain code-residual model with fixed anonymization (no counter leak)
# Changes from v1:
#   1. Per-example shuffled anonymization (no BM25 rank leak)
#   2. Neg sampling: 25/25/25/25 (added random negatives for calibration)
#   3. Output to v2 directory
#
# Requires: 1x GPU with ~18G free (bfloat16 + LoRA)
# Duration: ~7h on RTX 4090

set -euo pipefail

GPU=${1:-0}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUT=/data/chenlibin/grepo_agent_experiments/code_residual_7b_v2

mkdir -p $OUT

echo "Starting code-residual v2 training on GPU $GPU"
echo "Output: $OUT"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/train/train_rankft_code_residual.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data data/rankft/grepo_train.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir $OUT \
    --device cuda:0 \
    --num_negatives 4 \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --save_steps 50 \
    --code_max_lines 30 \
    --neg_samedir_ratio 0.25 --neg_pathdist_ratio 0.25 --neg_bm25_ratio 0.25 --neg_random_ratio 0.25 \
    --neg_graph_ratio 0.0 \
    --anonymize_paths \
    2>&1 | tee $OUT/train.log

echo "Training complete. Run eval with:"
echo "  bash scripts/rerun_main_results.sh"
