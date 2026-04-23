#!/bin/bash
set -euo pipefail
# Monitor delex50+cfneg combined training (GPU 7), launch eval when done

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

echo "=== Combined monitor started at $(date) ==="

# Wait for combined training to finish
while [ ! -f experiments/rankft_delex50_cfneg/final/adapter_model.safetensors ]; do
    sleep 300
done
echo "=== Combined training done at $(date) ==="

# Eval on graph pool
echo "Launching combined eval (graph pool) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_delex50_cfneg/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --graph_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/rankft_delex50_cfneg/eval_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "=== Combined graph eval done at $(date) ==="

# Eval on hybrid pool
echo "Launching combined eval (hybrid pool) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_delex50_cfneg/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --hybrid_candidates data/rankft/merged_hybrid_e5large_graph_candidates.jsonl \
    --output_dir experiments/rankft_delex50_cfneg/eval_hybrid \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "=== Combined hybrid eval done at $(date) ==="
