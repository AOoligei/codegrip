#!/bin/bash
# After 3B graph training on GPU 7: eval → bm25 training → bm25 eval
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
TRAIN_PID=2811825
GPU=7

echo "[$(date)] Waiting for 3B graph training (PID $TRAIN_PID)..."
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; done

if [ ! -d "experiments/scale_3B_graph/final" ]; then
    echo "[$(date)] ERROR: no final checkpoint for 3B graph"
    exit 1
fi

# 3B graph eval
echo "[$(date)] Running 3B graph eval on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --lora_path experiments/scale_3B_graph/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_3B_graph/eval_merged_rerank \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] 3B graph eval done."

# 3B bm25 training
echo "[$(date)] Starting 3B bm25 training on GPU $GPU..."
rm -rf experiments/scale_3B_bm25only 2>/dev/null
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/train/train_rankft.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --output_dir experiments/scale_3B_bm25only \
    --device cuda:0 --num_negatives 16 \
    --neg_bm25_ratio 1.0 --neg_graph_ratio 0.0 --neg_random_ratio 0.0 \
    --learning_rate 5e-5 --num_epochs 2 --batch_size 1 \
    --gradient_accumulation_steps 16 --save_steps 800 --logging_steps 10 \
    --max_seq_length 512 --lora_rank 32 --seed 42
echo "[$(date)] 3B bm25 training done."

# 3B bm25 eval
echo "[$(date)] Running 3B bm25 eval on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_rankft.py \
    --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
    --lora_path experiments/scale_3B_bm25only/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_3B_bm25only/eval_merged_rerank \
    --gpu_id 0 --top_k 200 --max_seq_length 512
echo "[$(date)] 3B bm25 eval done."
