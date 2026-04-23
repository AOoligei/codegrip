#!/bin/bash
# Master orchestration: monitors running jobs and launches new ones as GPUs free up
# Usage: nohup bash scripts/orchestrate_all.sh >> orchestrate.log 2>&1 &
set -e

BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] $1" | tee -a orchestrate.log; }

wait_for_pid() {
    local pid=$1
    local name=$2
    while kill -0 $pid 2>/dev/null; do
        sleep 60
    done
    log "$name (PID $pid) finished"
}

# Track which GPUs become free
# Current state:
# GPU 2: samedir eval (PID 3411331) — will finish first
# GPU 4: seed1 graph eval (PID 3493725) — will finish second
# GPU 3: seed1 bm25only training (PID 2890806) — ~1h left
# GPU 1: seed2 bm25only training (PID 3501294) — ~6h

log "Orchestrator started. Waiting for GPU 2 (samedir eval)..."

# === Phase 1: Wait for samedir eval, then use GPU 2 ===
wait_for_pid 3411331 "samedir eval"

# Fill samedir results
log "Running fill_structural_table.py..."
$PYTHON scripts/fill_structural_table.py

# Clean up failed seed2 graph run
rm -rf experiments/rankft_runB_graph_seed2/final experiments/rankft_runB_graph_seed2/best 2>/dev/null
log "Launching seed2 graph RE-TRAINING on GPU 2..."
$PYTHON src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/rankft_runB_graph_seed2 \
    --device cuda:2 \
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
    --lora_rank 32 \
    --seed 2 \
    2>&1 | tee experiments/rankft_runB_graph_seed2_v2.log
log "Seed2 graph training done on GPU 2"

# Eval seed2 graph on GPU 2
log "Evaluating seed2 graph on GPU 2..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph_seed2/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/rankft_runB_graph_seed2/eval_merged_rerank \
    --gpu_id 2 \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee experiments/rankft_runB_graph_seed2_eval.log
log "Seed2 graph eval done"

# === Phase 2: Wait for GPU 4 (seed1 graph eval), then use it ===
# By this point GPU 4 should be long free
# Run fair SWE-bench comparison on GPU 2 (after seed2 graph)
log "Running fair SWE-bench comparison on GPU 2..."
bash scripts/swebench_fair_comparison.sh 2

log "=== Phase 1+2 complete. Collecting seed results... ==="
$PYTHON scripts/collect_seed_results.py

log "Orchestrator complete."
