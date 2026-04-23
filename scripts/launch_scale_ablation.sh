#!/bin/bash
# Model scale ablation: graph-hard vs bm25-hard at different model sizes
# Tests whether graph-hard negative mining benefit is model-agnostic
# Usage: bash scripts/launch_scale_ablation.sh <model_size> <gpu_id>
# Example: bash scripts/launch_scale_ablation.sh 0.5B 0
set -e

SIZE=$1
GPU=$2
BASE=/home/chenlibin/grepo_agent
cd $BASE

# Model path mapping
case $SIZE in
    0.5B)
        MODEL_PATH=/data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct
        LORA_RANK=16
        ;;
    1.5B)
        MODEL_PATH=/data/chenlibin/models/Qwen2.5-1.5B-Instruct
        LORA_RANK=16
        ;;
    3B)
        MODEL_PATH=/data/chenlibin/models/Qwen2.5-3B-Instruct
        LORA_RANK=32
        ;;
    7B)
        MODEL_PATH=/data/shuyang/models/Qwen2.5-7B-Instruct
        LORA_RANK=32
        ;;
    14B)
        MODEL_PATH=/data/chenlibin/models/Qwen2.5-14B-Instruct
        LORA_RANK=32
        ;;
    *)
        echo "Unknown size: $SIZE. Use 0.5B, 1.5B, 3B, 7B, or 14B"
        exit 1
        ;;
esac

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python

echo "[$(date)] Starting scale ablation: ${SIZE} on GPU ${GPU}" | tee -a scale_ablation.log

# --- Graph-hard training ---
GRAPH_DIR=experiments/scale_${SIZE}_graph
echo "[$(date)] Training graph-hard ${SIZE}..." | tee -a scale_ablation.log

$PYTHON src/train/train_rankft.py \
    --model_path $MODEL_PATH \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir $GRAPH_DIR \
    --device cuda:$GPU \
    --num_negatives 16 \
    --neg_bm25_ratio 0.5 \
    --neg_graph_ratio 0.25 \
    --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 800 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank $LORA_RANK \
    --seed 42 \
    2>&1 | tee ${GRAPH_DIR}.log

echo "[$(date)] Graph-hard ${SIZE} done. Starting eval..." | tee -a scale_ablation.log

# --- Graph-hard eval ---
$PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL_PATH \
    --lora_path $GRAPH_DIR/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir $GRAPH_DIR/eval_merged_rerank \
    --gpu_id $GPU \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee ${GRAPH_DIR}_eval.log

echo "[$(date)] Graph-hard ${SIZE} eval done." | tee -a scale_ablation.log

# --- BM25-only training ---
BM25_DIR=experiments/scale_${SIZE}_bm25only
echo "[$(date)] Training bm25-only ${SIZE}..." | tee -a scale_ablation.log

$PYTHON src/train/train_rankft.py \
    --model_path $MODEL_PATH \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir $BM25_DIR \
    --device cuda:$GPU \
    --num_negatives 16 \
    --neg_bm25_ratio 1.0 \
    --neg_graph_ratio 0.0 \
    --neg_random_ratio 0.0 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 800 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank $LORA_RANK \
    --seed 42 \
    2>&1 | tee ${BM25_DIR}.log

echo "[$(date)] BM25-only ${SIZE} done. Starting eval..." | tee -a scale_ablation.log

# --- BM25-only eval ---
$PYTHON src/eval/eval_rankft.py \
    --model_path $MODEL_PATH \
    --lora_path $BM25_DIR/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir $BM25_DIR/eval_merged_rerank \
    --gpu_id $GPU \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee ${BM25_DIR}_eval.log

echo "[$(date)] Scale ablation ${SIZE} COMPLETE." | tee -a scale_ablation.log
