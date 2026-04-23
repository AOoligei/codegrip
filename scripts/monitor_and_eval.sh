#!/bin/bash
set -euo pipefail
# Monitor training experiments and auto-evaluate when they finish
# Usage: nohup bash scripts/monitor_and_eval.sh [gpu_id] &

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"
GPU="${1:-6}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

check_and_eval() {
    local EXP_NAME=$1
    local MODEL_PATH=$2
    local ADAPTER_DIR=$3

    # Check if training is done (final checkpoint exists)
    if [ -d "$ADAPTER_DIR" ] && [ -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
        # Check if already evaluated
        if [ -f "experiments/${EXP_NAME}/eval_reranked/predictions.jsonl" ]; then
            return 0  # Already done
        fi
        echo "[$(date)] $EXP_NAME: Training complete, running evaluation pipeline..."
        bash scripts/eval_all_with_expansion.sh "$GPU" 2>&1 | tee "experiments/${EXP_NAME}/full_eval.log"
        return 0
    fi
    return 1  # Not done yet
}

MODEL_QWEN="/data/shuyang/models/Qwen2.5-7B-Instruct"
MODEL_CODER="/home/chenlibin/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"

echo "[$(date)] Starting training monitor..."

while true; do
    all_done=true
    
    # Check exp5
    if ! check_and_eval "exp5_coder_sft_only" "$MODEL_CODER" "experiments/exp5_coder_sft_only/stage2_sft/final"; then
        all_done=false
    fi
    
    # Check exp6
    if ! check_and_eval "exp6_warmstart_cochange" "$MODEL_QWEN" "experiments/exp6_warmstart_cochange/stage2_sft/final"; then
        all_done=false
    fi
    
    # Check exp7
    if ! check_and_eval "exp7_multitask_sft" "$MODEL_QWEN" "experiments/exp7_multitask_sft/stage2_sft/final"; then
        all_done=false
    fi

    # Check exp8 (graph-conditioned SFT) — has its own eval script
    if [ -f "experiments/exp8_graph_sft/stage2_sft/final/adapter_model.safetensors" ] && \
       [ ! -f "experiments/exp8_graph_sft/eval_graph_reranked/predictions.jsonl" ]; then
        echo "[$(date)] exp8: Training complete, running graph-mode evaluation..."
        bash scripts/eval_exp8.sh "$GPU" 2>&1 | tee "experiments/exp8_graph_sft/full_eval.log"
    fi

    # Check exp9 (TGS filetree)
    if ! check_and_eval "exp9_tgs_filetree" "$MODEL_QWEN" "experiments/exp9_tgs_filetree/stage2_sft/final"; then
        all_done=false
    fi

    # Check exp10 (TGS+graph) — has its own eval script
    if [ -f "experiments/exp10_tgs_graph/stage2_sft/final/adapter_model.safetensors" ] && \
       [ ! -f "experiments/exp10_tgs_graph/eval_graph_reranked/predictions.jsonl" ]; then
        echo "[$(date)] exp10: Training complete, running graph-mode evaluation..."
        bash scripts/eval_exp10.sh "$GPU" 2>&1 | tee "experiments/exp10_tgs_graph/full_eval.log"
    fi

    if $all_done; then
        echo "[$(date)] All experiments evaluated! Running ensemble..."
        # Run ensemble of all available models
        PRED_FILES=""
        for exp in exp1_sft_only exp5_coder_sft_only exp6_warmstart_cochange exp7_multitask_sft exp9_tgs_filetree; do
            f="experiments/$exp/eval_filetree/predictions.jsonl"
            if [ -f "$f" ]; then
                PRED_FILES="$PRED_FILES $f"
            fi
        done
        # Also include graph-mode predictions
        for exp in exp8_graph_sft exp10_tgs_graph; do
            f="experiments/$exp/eval_graph/predictions.jsonl"
            if [ -f "$f" ]; then
                PRED_FILES="$PRED_FILES $f"
            fi
        done
        
        if [ -n "$PRED_FILES" ]; then
            echo "Ensembling: $PRED_FILES"
            $PYTHON src/eval/ensemble_predictions.py \
                --predictions $PRED_FILES \
                --method rrf \
                --output experiments/ensemble_rrf/predictions.jsonl
            
            # Expand and rerank ensemble
            $PYTHON src/eval/multi_signal_expansion.py \
                --predictions experiments/ensemble_rrf/predictions.jsonl \
                --train_data data/grepo_text/grepo_train.jsonl \
                --dep_graph_dir data/dep_graphs \
                --file_tree_dir data/file_trees \
                --output experiments/ensemble_rrf/eval_unified_expansion/predictions.jsonl \
                --max_expand 35 --min_cochange_score 0.02 --max_dir_size 35
            
            $PYTHON src/eval/rerank_predictions.py \
                --predictions experiments/ensemble_rrf/eval_unified_expansion/predictions.jsonl \
                --test_data data/grepo_text/grepo_test.jsonl \
                --output experiments/ensemble_rrf/eval_reranked/predictions.jsonl \
                --promote_top5 1 --promote_top10 3 --threshold 0.2
        fi
        
        echo "[$(date)] All done!"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done
