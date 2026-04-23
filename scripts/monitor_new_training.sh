#!/bin/bash
set -euo pipefail
# Monitor 3B delex50 (GPU 4), combined (GPU 7), seed2 (GPU 1)
# Launch evals as training completes

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

echo "=== Monitor started at $(date) ==="

# Wait for 3B delex50 to finish
while [ ! -f experiments/scale_3B_delex50/final/adapter_model.safetensors ]; do
    sleep 300
done
echo "=== 3B delex50 training done at $(date) ==="

# Launch 3B delex50 eval on graph pool (reuse GPU 4)
echo "Launching 3B delex50 eval (graph pool) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --lora_path experiments/scale_3B_delex50/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --graph_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_3B_delex50/eval_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "=== 3B delex50 graph eval done at $(date) ==="

# Launch 3B delex50 eval on hybrid pool
echo "Launching 3B delex50 eval (hybrid pool) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --lora_path experiments/scale_3B_delex50/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --hybrid_candidates data/rankft/merged_hybrid_e5large_graph_candidates.jsonl \
    --output_dir experiments/scale_3B_delex50/eval_hybrid \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "=== 3B delex50 hybrid eval done at $(date) ==="
echo "All 3B delex50 evals complete."
