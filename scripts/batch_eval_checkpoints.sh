#!/bin/bash
# Eval ALL checkpoints for a completed experiment, pick best
# Usage: bash scripts/batch_eval_checkpoints.sh <exp_dir> <model_path> <gpu_id> [candidates2]
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"

EXP_DIR=$1
MODEL_PATH=$2
GPU=$3
CANDS2=${4:-""}

TEST_DATA="data/swebench_lite/swebench_lite_test.jsonl"
CANDS1="data/rankft/swebench_test_bm25_top500.jsonl"

get_h1() {
    $PYTHON -c "import json; print(f\"{json.load(open('$1'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "0.00"
}

BEST_H1="0.00"
BEST_CKPT=""
BEST_H1T="0.00"
BEST_CKPT_T=""

# Collect all checkpoints
CKPTS=$(ls -d ${EXP_DIR}/checkpoint-* ${EXP_DIR}/best ${EXP_DIR}/final 2>/dev/null | while read d; do
    [ -f "$d/adapter_config.json" ] && echo "$d"
done)

TOTAL=$(echo "$CKPTS" | wc -l)
echo "[$(date)] Evaluating ${TOTAL} checkpoints in ${EXP_DIR} on GPU ${GPU}"

i=0
for ckpt in $CKPTS; do
    i=$((i+1))
    NAME=$(basename "$ckpt")
    echo "[${i}/${TOTAL}] ${NAME}..."
    
    # Eval on BM25
    OUTDIR="${EXP_DIR}/eval_${NAME}_bm25"
    CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
        --model_path ${MODEL_PATH} --lora_path ${ckpt} \
        --test_data ${TEST_DATA} --bm25_candidates ${CANDS1} \
        --output_dir ${OUTDIR} --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
        2>/dev/null
    
    H1=$(get_h1 "${OUTDIR}/summary.json")
    
    # Eval on tricked if provided
    H1T="N/A"
    if [ -n "$CANDS2" ]; then
        OUTDIR_T="${EXP_DIR}/eval_${NAME}_tricked"
        CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
            --model_path ${MODEL_PATH} --lora_path ${ckpt} \
            --test_data ${TEST_DATA} --bm25_candidates ${CANDS2} \
            --output_dir ${OUTDIR_T} --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
            2>/dev/null
        H1T=$(get_h1 "${OUTDIR_T}/summary.json")
    fi
    
    # Track best
    IS_BEST=""
    if $PYTHON -c "exit(0 if float('${H1}') > float('${BEST_H1}') else 1)" 2>/dev/null; then
        BEST_H1="$H1"; BEST_CKPT="$NAME"; IS_BEST=" *** BEST ***"
    fi
    if [ "$H1T" != "N/A" ] && $PYTHON -c "exit(0 if float('${H1T}') > float('${BEST_H1T}') else 1)" 2>/dev/null; then
        BEST_H1T="$H1T"; BEST_CKPT_T="$NAME"
    fi
    
    echo "  ${NAME}: BM25=${H1}% Tricked=${H1T}%${IS_BEST}"
    
    # Delete non-best checkpoint to save disk (keep best, final, and current best)
    if [ "$NAME" != "best" ] && [ "$NAME" != "final" ] && [ "$NAME" != "$BEST_CKPT" ]; then
        rm -rf "$ckpt"
        rm -rf "${EXP_DIR}/eval_${NAME}_bm25" "${EXP_DIR}/eval_${NAME}_tricked" 2>/dev/null
    fi
done

echo ""
echo "=========================================="
echo "RESULT: ${EXP_DIR}"
echo "  Best BM25:    ${BEST_H1}% (${BEST_CKPT})"
echo "  Best Tricked: ${BEST_H1T}% (${BEST_CKPT_T})"
echo "=========================================="

# Save result
$PYTHON -c "
import json
json.dump({
    'best_bm25_h1': float('${BEST_H1}'), 'best_bm25_ckpt': '${BEST_CKPT}',
    'best_tricked_h1': float('${BEST_H1T}'), 'best_tricked_ckpt': '${BEST_CKPT_T}'
}, open('${EXP_DIR}/best_checkpoint_result.json', 'w'), indent=2)
"
