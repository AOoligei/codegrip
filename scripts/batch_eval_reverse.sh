#!/bin/bash
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

# Collect checkpoints in REVERSE order
CKPTS=$(ls -d ${EXP_DIR}/checkpoint-* ${EXP_DIR}/best ${EXP_DIR}/final 2>/dev/null | while read d; do
    [ -f "$d/adapter_config.json" ] && echo "$d"
done | tac)

TOTAL=$(echo "$CKPTS" | wc -l)
echo "[REVERSE] Evaluating ${TOTAL} checkpoints in ${EXP_DIR} on GPU ${GPU}"

BEST_H1="0.00"
BEST_CKPT=""
i=0

for ckpt in $CKPTS; do
    i=$((i+1))
    NAME=$(basename "$ckpt")
    
    # Skip if forward pass already did this one
    [ -f "${EXP_DIR}/eval_${NAME}_bm25/summary.json" ] && continue
    
    OUTDIR="${EXP_DIR}/eval_${NAME}_bm25"
    CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
        --model_path ${MODEL_PATH} --lora_path ${ckpt} \
        --test_data ${TEST_DATA} --bm25_candidates ${CANDS1} \
        --output_dir ${OUTDIR} --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
        2>/dev/null
    
    H1=$(get_h1 "${OUTDIR}/summary.json")
    
    H1T="N/A"
    if [ -n "$CANDS2" ]; then
        OUTDIR_T="${EXP_DIR}/eval_${NAME}_tricked"
        [ -f "${OUTDIR_T}/summary.json" ] && { H1T=$(get_h1 "${OUTDIR_T}/summary.json"); } || {
            CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
                --model_path ${MODEL_PATH} --lora_path ${ckpt} \
                --test_data ${TEST_DATA} --bm25_candidates ${CANDS2} \
                --output_dir ${OUTDIR_T} --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
                2>/dev/null
            H1T=$(get_h1 "${OUTDIR_T}/summary.json")
        }
    fi
    
    IS_BEST=""
    if $PYTHON -c "exit(0 if float('${H1}') > float('${BEST_H1}') else 1)" 2>/dev/null; then
        BEST_H1="$H1"; BEST_CKPT="$NAME"; IS_BEST=" ***"
    fi
    echo "  [R ${i}/${TOTAL}] ${NAME}: BM25=${H1}% Tricked=${H1T}%${IS_BEST}"
done
echo "[REVERSE] Done"
