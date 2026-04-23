#!/bin/bash
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE
log() { echo "[$(date)] [GPU5-SEED4] $1" | tee -a orchestrate.log; }

SEED=4

# Graph-hard seed4
EXP_G="experiments/rankft_runB_graph_seed${SEED}"
if [ ! -d "${EXP_G}/final" ]; then
    log "Training graph-hard seed${SEED} on GPU 5..."
    $PYTHON src/train/train_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --dep_graph_dir data/dep_graphs \
        --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
        --file_tree_dir data/file_trees \
        --output_dir ${EXP_G} \
        --device cuda:5 --num_negatives 16 \
        --neg_bm25_ratio 0.5 --neg_graph_ratio 0.25 --neg_random_ratio 0.25 \
        --learning_rate 5e-5 --num_epochs 2 --batch_size 1 \
        --gradient_accumulation_steps 16 --save_steps 800 --logging_steps 10 \
        --max_seq_length 512 --lora_rank 32 --seed ${SEED} \
        2>&1 | tee ${EXP_G}_train.log
    log "Graph-hard seed${SEED} training done."
fi

# Eval graph
log "Evaluating graph-hard seed${SEED}..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path ${EXP_G}/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir ${EXP_G}/eval_merged_rerank \
    --gpu_id 5 --top_k 200 --max_seq_length 512 \
    2>&1 | tee ${EXP_G}_eval.log
log "Graph-hard seed${SEED} eval done."

# BM25-only seed4
EXP_B="experiments/rankft_runA_bm25only_seed${SEED}"
if [ ! -d "${EXP_B}/final" ]; then
    log "Training bm25-only seed${SEED} on GPU 5..."
    $PYTHON src/train/train_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --output_dir ${EXP_B} \
        --device cuda:5 --num_negatives 16 \
        --neg_bm25_ratio 1.0 --neg_graph_ratio 0.0 --neg_random_ratio 0.0 \
        --learning_rate 5e-5 --num_epochs 2 --batch_size 1 \
        --gradient_accumulation_steps 16 --save_steps 800 --logging_steps 10 \
        --max_seq_length 512 --lora_rank 32 --seed ${SEED} \
        2>&1 | tee ${EXP_B}_train.log
    log "BM25-only seed${SEED} training done."
fi

# Eval bm25
log "Evaluating bm25-only seed${SEED}..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path ${EXP_B}/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir ${EXP_B}/eval_merged_rerank \
    --gpu_id 5 --top_k 200 --max_seq_length 512 \
    2>&1 | tee ${EXP_B}_eval.log
log "BM25-only seed${SEED} eval done."

$PYTHON scripts/collect_seed_results.py
log "GPU5 seed4 COMPLETE."
