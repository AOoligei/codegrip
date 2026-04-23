#!/bin/bash
set -e
cd /home/chenlibin/grepo_agent
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3

echo "Starting bi-encoder eval at $(date)"

for perturb in none shuffle_filenames shuffle_dirs; do
    echo ""
    echo "=========================================="
    echo "Running perturb=${perturb} at $(date)"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=6 $PY scripts/eval_biencoder_reranker.py --perturb $perturb --gpu_id 0
done

echo ""
echo "All done at $(date)"
