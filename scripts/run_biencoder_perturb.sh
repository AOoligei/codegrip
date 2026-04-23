#!/bin/bash
set -e
cd /home/chenlibin/grepo_agent

echo "=== shuffle_filenames ==="
CUDA_VISIBLE_DEVICES=6 /home/chenlibin/miniconda3/envs/tgn/bin/python3 scripts/eval_biencoder_reranker.py --perturb shuffle_filenames --gpu_id 0

echo ""
echo "=== shuffle_dirs ==="
CUDA_VISIBLE_DEVICES=6 /home/chenlibin/miniconda3/envs/tgn/bin/python3 scripts/eval_biencoder_reranker.py --perturb shuffle_dirs --gpu_id 0
