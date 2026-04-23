#!/bin/bash
# Recovery: seed3 and seed4 bm25-only training + eval on GPU 1
set -eo pipefail
GPU=1
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [SEEDS-BM25] $1" | tee -a orchestrate.log; }

# Wait for GPU 1 to be free
log "Waiting for GPU 1 to be free..."
while nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
done
log "GPU 1 is free."

for SEED in 3 4; do
    EXP="experiments/rankft_runA_bm25only_seed${SEED}"

    # Clean up failed attempt
    rm -rf "${EXP}" 2>/dev/null

    # Train
    log "Training bm25-only seed${SEED} on GPU ${GPU}..."
    $PYTHON src/train/train_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/exp1_sft_only/stage2_sft/final \
        --train_data data/grepo_text/grepo_train.jsonl \
        --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
        --output_dir ${EXP} \
        --device cuda:${GPU} \
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
        --lora_rank 32 \
        --seed ${SEED} \
        2>&1 | tee ${EXP}_train.log
    log "BM25-only seed${SEED} training done."

    # Eval
    log "Evaluating bm25-only seed${SEED}..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path ${EXP}/final \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir ${EXP}/eval_merged_rerank \
        --gpu_id ${GPU} \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee ${EXP}_eval.log
    log "BM25-only seed${SEED} eval done."
done

# Collect results
log "Collecting seed results..."
$PYTHON scripts/collect_seed_results.py

log "Seeds BM25 recovery complete."
