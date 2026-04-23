#!/bin/bash
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"
DIR="experiments/rankft_runB_graph_v2"
TEST="data/grepo_text/grepo_test.jsonl"
GPU=$1; CAND=$2; POOL=$3

CKPTS=$(ls -d ${DIR}/checkpoint-* ${DIR}/best ${DIR}/final 2>/dev/null | while read d; do
    [ -f "$d/adapter_config.json" ] && echo "$d"
done | tac)  # REVERSE order
TOTAL=$(echo "$CKPTS" | wc -l)
BEST_H1="0.00"; BEST_CKPT=""; i=0
for ckpt in $CKPTS; do
    i=$((i+1)); NAME=$(basename "$ckpt")
    OUTDIR="${DIR}/eval_${NAME}_${POOL}"
    [ -f "${OUTDIR}/summary.json" ] && continue
    CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
        --model_path ${MODEL} --lora_path ${ckpt} --test_data ${TEST} \
        --bm25_candidates ${CAND} --output_dir ${OUTDIR} \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 1 2>/dev/null
    H1=$($PYTHON -c "import json; print(f\"{json.load(open('${OUTDIR}/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "0.00")
    IS_BEST=""
    $PYTHON -c "exit(0 if float('${H1}') > float('${BEST_H1}') else 1)" 2>/dev/null && { BEST_H1="$H1"; BEST_CKPT="$NAME"; IS_BEST=" ***"; }
    echo "[R ${i}/${TOTAL}] ${NAME} ${POOL}: ${H1}%${IS_BEST}"
done
echo ">>> REV BEST ${POOL}: ${BEST_H1}% (${BEST_CKPT})"
