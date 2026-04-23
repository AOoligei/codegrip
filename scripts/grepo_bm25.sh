#!/bin/bash
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"
DIR="experiments/rankft_runB_graph_v2"
TEST="data/grepo_text/grepo_test.jsonl"
CAND="data/rankft/grepo_test_bm25_top500.jsonl"
GPU=$1

for ckpt in best final checkpoint-50 checkpoint-100 checkpoint-150 checkpoint-200 checkpoint-300 checkpoint-400 checkpoint-500 checkpoint-600 checkpoint-700; do
    CKPT_DIR="${DIR}/${ckpt}"
    [ -f "${CKPT_DIR}/adapter_config.json" ] || continue
    OUT="${DIR}/eval_${ckpt}_bm25"
    [ -f "${OUT}/summary.json" ] && continue
    CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
        --model_path ${MODEL} --lora_path ${CKPT_DIR} --test_data ${TEST} \
        --bm25_candidates ${CAND} --output_dir ${OUT} \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 4 2>/dev/null
    H1=$($PYTHON -c "import json; print(f\"{json.load(open('${OUT}/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    echo "${ckpt} bm25: ${H1}%"
done
echo "DONE"
