#!/bin/bash
# Run SweRank path perturbation experiments.
#
# Tests whether Salesforce's SOTA SweRankEmbed model exhibits path-prior bias
# by running on our SWE-bench Lite perturbation data.
#
# Usage:
#   bash scripts/run_swerank_perturb.sh GPU_ID
#   # e.g., bash scripts/run_swerank_perturb.sh 5
#
# Prerequisites:
#   1. Download models first (run once):
#      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Salesforce/SweRankEmbed-Small', cache_folder='/data/chenlibin/models', trust_remote_code=True)"
#      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Salesforce/SweRankEmbed-Large', cache_folder='/data/chenlibin/models', trust_remote_code=True)"
#   2. Ensure swebench perturbation data exists in data/swebench_lite/

set -e
cd /home/chenlibin/grepo_agent

GPU_ID=${1:-5}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUTPUT_DIR=experiments/swerank_perturb

echo "============================================"
echo "SweRank Path Perturbation Experiments"
echo "GPU: ${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

mkdir -p ${OUTPUT_DIR}

# ── SweRankEmbed-Small (137M) ──────────────────────────────────────────
echo ""
echo ">>> SweRankEmbed-Small: none (baseline)"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model small --perturb none --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/small_none.log

echo ""
echo ">>> SweRankEmbed-Small: shuffle_dirs"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model small --perturb shuffle_dirs --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/small_shuffle_dirs.log

echo ""
echo ">>> SweRankEmbed-Small: shuffle_filenames"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model small --perturb shuffle_filenames --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/small_shuffle_filenames.log

echo ""
echo ">>> SweRankEmbed-Small: flatten_dirs"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model small --perturb flatten_dirs --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/small_flatten_dirs.log

# ── SweRankEmbed-Large (7B) ───────────────────────────────────────────
echo ""
echo ">>> SweRankEmbed-Large: none (baseline)"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model large --perturb none --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/large_none.log

echo ""
echo ">>> SweRankEmbed-Large: shuffle_dirs"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model large --perturb shuffle_dirs --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/large_shuffle_dirs.log

echo ""
echo ">>> SweRankEmbed-Large: shuffle_filenames"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model large --perturb shuffle_filenames --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/large_shuffle_filenames.log

echo ""
echo ">>> SweRankEmbed-Large: flatten_dirs"
CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} scripts/eval_swerank_perturb.py \
    --model large --perturb flatten_dirs --gpu_id 0 --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/large_flatten_dirs.log

echo ""
echo "============================================"
echo "All experiments complete. Results in ${OUTPUT_DIR}/"
echo "============================================"
