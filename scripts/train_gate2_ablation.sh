#!/bin/bash
# Gate 2 codeaware ablation matrix.
# Each run: 7B Qwen2.5-7B + LoRA, code prompt, seed 42.
#
# Variants (TRAIN_POOL × AUGMENT):
#   1. SWE-only + augment=0.0  (noaug baseline)     → tag: SWEonly_aug0
#   2. SWE-only + augment=0.5  (existing, skip)     → use experiments/rankft_codeaware_swetrain/best
#   3. SWE+GREPO + augment=0.5                      → tag: SWEGRepo_aug05
#   4. GREPO-only + augment=0.5                     → tag: GRepoonly_aug05
#
# Usage: bash train_gate2_ablation.sh <TAG> <TRAIN_DATA> <BM25_CANDS> <REPO_DIR> <AUGMENT> <GPU_ID>
set -e
TAG=${1:?tag}
TRAIN_DATA=${2:?train}
BM25=${3:?bm25}
REPO_DIR=${4:?repo_dir}
AUGMENT=${5:-0.5}
GPU=${6:-0}
OUT=/data/chenlibin/grepo_agent_experiments/gate2_${TAG}

cd /home/chenlibin/grepo_agent
mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPU /home/chenlibin/miniconda3/envs/tgn/bin/python3 -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --train_data "$TRAIN_DATA" \
    --bm25_candidates "$BM25" \
    --repo_dir "$REPO_DIR" \
    --output_dir "$OUT" \
    --include_code \
    --code_max_lines 50 \
    --path_augment_fraction $AUGMENT \
    --num_negatives 8 \
    --neg_bm25_ratio 1.0 \
    --neg_random_ratio 0.0 \
    --neg_graph_ratio 0.0 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --logging_steps 10 \
    --max_seq_length 768 \
    --lora_rank 32 \
    --seed 42 \
    2>&1 | tee logs/gate2_${TAG}.log
