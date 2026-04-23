#!/bin/bash
# CodeGRIP experiment runner
# Orchestrates all experiments for the paper
#
# Experiment plan:
# Exp 0: Zero-shot baseline (already done)
# Exp 1: SFT-only (file tree context, no graph pre-training)
# Exp 2: GSP + SFT (CodeGRIP two-stage, co-change GSP)
# Exp 3: AST-GSP + SFT (CodeGRIP two-stage, AST-based GSP)
# Exp 4: Combined GSP + SFT (both co-change + AST)
#
# Each experiment trains on GPU and evaluates on test set

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

# Base model (use Coder model if available, else general)
CODER_MODEL="/home/chenlibin/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots"
GENERAL_MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"

# Check if Coder model is available
if [ -d "$CODER_MODEL" ] && [ "$(ls $CODER_MODEL)" ]; then
    MODEL_PATH="$CODER_MODEL/$(ls $CODER_MODEL | head -1)"
    echo "Using Qwen2.5-Coder-7B-Instruct: $MODEL_PATH"
else
    MODEL_PATH="$GENERAL_MODEL"
    echo "Coder model not available, using Qwen2.5-7B-Instruct: $MODEL_PATH"
fi

# Common training args
LORA_RANK=32
LORA_ALPHA=64
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_SEQ=4096

# ============================================================
# Exp 1: SFT-only baseline
# ============================================================
run_exp1() {
    local GPU=$1
    echo "=== Exp 1: SFT-only (filetree) on GPU $GPU ==="
    $PYTHON src/train/train_codegrip.py \
        --model_path "$MODEL_PATH" \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/exp1_sft_only \
        --skip_gsp \
        --sft_epochs 3 --sft_lr 1e-4 \
        --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
        --max_seq_length $MAX_SEQ \
        --device "cuda:$GPU" 2>&1 | tee experiments/exp1_sft_only/train.log
}

# ============================================================
# Exp 2: Co-change GSP + SFT
# ============================================================
run_exp2() {
    local GPU=$1
    echo "=== Exp 2: Co-change GSP + SFT on GPU $GPU ==="
    $PYTHON src/train/train_codegrip.py \
        --model_path "$MODEL_PATH" \
        --gsp_data data/gsp/gsp_all.jsonl \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/exp2_cochange_gsp_sft \
        --gsp_epochs 2 --gsp_lr 2e-4 \
        --sft_epochs 3 --sft_lr 1e-4 \
        --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
        --max_seq_length $MAX_SEQ \
        --device "cuda:$GPU" 2>&1 | tee experiments/exp2_cochange_gsp_sft/train.log
}

# ============================================================
# Exp 3: AST-GSP + SFT
# ============================================================
run_exp3() {
    local GPU=$1
    echo "=== Exp 3: AST GSP + SFT on GPU $GPU ==="
    $PYTHON src/train/train_codegrip.py \
        --model_path "$MODEL_PATH" \
        --gsp_data data/gsp_ast/gsp_ast_all.jsonl \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/exp3_ast_gsp_sft \
        --gsp_epochs 2 --gsp_lr 2e-4 \
        --sft_epochs 3 --sft_lr 1e-4 \
        --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
        --max_seq_length $MAX_SEQ \
        --device "cuda:$GPU" 2>&1 | tee experiments/exp3_ast_gsp_sft/train.log
}

# ============================================================
# Exp 4: Combined (co-change + AST) GSP + SFT
# ============================================================
run_exp4() {
    local GPU=$1
    echo "=== Exp 4: Combined GSP + SFT on GPU $GPU ==="

    # First combine GSP data
    cat data/gsp/gsp_all.jsonl data/gsp_ast/gsp_ast_all.jsonl > data/gsp_combined.jsonl
    echo "Combined GSP: $(wc -l < data/gsp_combined.jsonl) examples"

    $PYTHON src/train/train_codegrip.py \
        --model_path "$MODEL_PATH" \
        --gsp_data data/gsp_combined.jsonl \
        --sft_data data/sft/sft_v1_filetree.jsonl \
        --output_dir experiments/exp4_combined_gsp_sft \
        --gsp_epochs 2 --gsp_lr 2e-4 \
        --sft_epochs 3 --sft_lr 1e-4 \
        --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
        --max_seq_length $MAX_SEQ \
        --device "cuda:$GPU" 2>&1 | tee experiments/exp4_combined_gsp_sft/train.log
}

# ============================================================
# Evaluation helper
# ============================================================
run_eval() {
    local EXP_NAME=$1
    local ADAPTER_PATH=$2
    local GPU=$3
    echo "=== Evaluating $EXP_NAME on GPU $GPU ==="

    $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$ADAPTER_PATH" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir "experiments/${EXP_NAME}/eval_results" \
        --prompt_mode filetree \
        --file_tree_dir data/file_trees \
        --device "cuda:$GPU" 2>&1 | tee "experiments/${EXP_NAME}/eval.log"
}

# ============================================================
# Main
# ============================================================
case "${1:-all}" in
    exp1) run_exp1 "${2:-3}" ;;
    exp2) run_exp2 "${2:-4}" ;;
    exp3) run_exp3 "${2:-5}" ;;
    exp4) run_exp4 "${2:-6}" ;;
    eval)
        EXP="${2:-exp1_sft_only}"
        run_eval "$EXP" "experiments/$EXP/stage2_sft/final" "${3:-2}"
        ;;
    all)
        echo "Running all experiments in parallel on GPUs 3-6..."
        mkdir -p experiments/{exp1_sft_only,exp2_cochange_gsp_sft,exp3_ast_gsp_sft,exp4_combined_gsp_sft}
        run_exp1 3 &
        run_exp2 4 &
        run_exp3 5 &
        run_exp4 6 &
        wait
        echo "All training complete! Running evaluations..."
        run_eval exp1_sft_only "experiments/exp1_sft_only/stage2_sft/final" 3 &
        run_eval exp2_cochange_gsp_sft "experiments/exp2_cochange_gsp_sft/stage2_sft/final" 4 &
        run_eval exp3_ast_gsp_sft "experiments/exp3_ast_gsp_sft/stage2_sft/final" 5 &
        run_eval exp4_combined_gsp_sft "experiments/exp4_combined_gsp_sft/stage2_sft/final" 6 &
        wait
        echo "All done!"
        ;;
    *)
        echo "Usage: $0 {exp1|exp2|exp3|exp4|eval|all} [gpu_id]"
        ;;
esac
