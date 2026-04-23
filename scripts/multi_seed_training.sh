#!/bin/bash
set -euo pipefail
# Multi-seed training for path-only reranker (train_rankft.py).
# Trains with seeds 42, 123, 456 sequentially on a single GPU.
# Seed 42 is symlinked from the existing rankft_runB_graph run.
#
# Usage: bash scripts/multi_seed_training.sh GPU_ID
#   e.g.: nohup bash scripts/multi_seed_training.sh 4 > logs/multiseed.log 2>&1 &

cd /home/chenlibin/grepo_agent

GPU=${1:?Usage: $0 GPU_ID}
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_INIT="experiments/exp1_sft_only/stage2_sft/final"
TRAIN_DATA="data/rankft/grepo_train.jsonl"
BM25_TRAIN="data/rankft/grepo_train_bm25_top500.jsonl"
DEP_GRAPH_DIR="data/dep_graphs"
COCHANGE_DATA="data/grepo_text/grepo_train.jsonl"
FILE_TREE_DIR="data/file_trees"
TEST_DATA="data/grepo_text/grepo_test.jsonl"
GRAPH_CANDIDATES="data/rankft/merged_bm25_exp6_candidates.jsonl"

BASE_OUT="/data/chenlibin/grepo_agent_experiments"
mkdir -p "$BASE_OUT"

SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    OUTDIR="${BASE_OUT}/multiseed_seed${SEED}"
    echo "=============================================="
    echo "=== Seed ${SEED} — started at $(date) ==="
    echo "=============================================="

    # ---- Training ----
    if [ "$SEED" -eq 42 ]; then
        # Seed 42 already trained as experiments/rankft_runB_graph.
        # Symlink instead of retraining.
        if [ ! -e "${OUTDIR}" ]; then
            echo "Symlinking seed 42 from existing run..."
            ln -sfn /home/chenlibin/grepo_agent/experiments/rankft_runB_graph "${OUTDIR}"
        else
            echo "Seed 42 output already exists at ${OUTDIR}, skipping training."
        fi
    else
        if [ -f "${OUTDIR}/final/adapter_model.safetensors" ]; then
            echo "Seed ${SEED} already trained (final adapter found), skipping training."
        else
            echo "Training seed ${SEED}..."
            CUDA_VISIBLE_DEVICES=${GPU} $PYTHON -u src/train/train_rankft.py \
                --model_path ${MODEL} \
                --lora_path ${LORA_INIT} \
                --train_data ${TRAIN_DATA} \
                --bm25_candidates ${BM25_TRAIN} \
                --dep_graph_dir ${DEP_GRAPH_DIR} \
                --train_data_for_cochange ${COCHANGE_DATA} \
                --file_tree_dir ${FILE_TREE_DIR} \
                --output_dir ${OUTDIR} \
                --device cuda:0 \
                --num_negatives 16 \
                --neg_bm25_ratio 0.5 \
                --neg_graph_ratio 0.25 \
                --neg_random_ratio 0.25 \
                --learning_rate 5e-5 \
                --num_epochs 2 \
                --batch_size 1 \
                --gradient_accumulation_steps 16 \
                --save_steps 200 \
                --logging_steps 10 \
                --max_seq_length 512 \
                --lora_rank 32 \
                --seed ${SEED}
        fi
    fi
    echo "=== Seed ${SEED} training done at $(date) ==="

    # ---- Evaluation on graph-expanded pool ----
    EVAL_DIR="${OUTDIR}/eval_graph_rerank"

    # Determine LoRA path: use best if available, else final
    if [ -f "${OUTDIR}/best/adapter_model.safetensors" ]; then
        LORA_EVAL="${OUTDIR}/best"
    elif [ -f "${OUTDIR}/final/adapter_model.safetensors" ]; then
        LORA_EVAL="${OUTDIR}/final"
    else
        echo "ERROR: No adapter found for seed ${SEED}, skipping eval."
        continue
    fi

    # Check if eval already exists — ONLY check eval_graph_rerank (our standardized dir)
    # Do NOT reuse old evals that may have used different candidate files
    EVAL_FOUND=false
    if [ -f "${EVAL_DIR}/summary.json" ]; then
        echo "Eval already done for seed ${SEED} (${EVAL_DIR}), skipping."
        EVAL_FOUND=true
    fi

    if [ "$EVAL_FOUND" = false ]; then
        echo "Evaluating seed ${SEED} (lora: ${LORA_EVAL})..."
        CUDA_VISIBLE_DEVICES=${GPU} $PYTHON -u scripts/eval_rankft_4bit.py \
            --model_path ${MODEL} \
            --lora_path ${LORA_EVAL} \
            --test_data ${TEST_DATA} \
            --graph_candidates ${GRAPH_CANDIDATES} \
            --output_dir ${EVAL_DIR} \
            --gpu_id 0 \
            --top_k 200 \
            --max_seq_length 512 \
            --score_batch_size 16
    fi
    echo "=== Seed ${SEED} eval done at $(date) ==="
done

echo ""
echo "=============================================="
echo "All seeds complete. Collecting results..."
echo "=============================================="
$PYTHON scripts/collect_multiseed_results.py \
    --base_dir "${BASE_OUT}" \
    --seeds 42 123 456
