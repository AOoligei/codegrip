#!/bin/bash
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"
DIR="experiments/rankft_runB_graph_v2"
TEST="data/grepo_text/grepo_test.jsonl"
GRAPH_CAND="data/rankft/merged_bm25_exp6_candidates.jsonl"
HYBRID_CAND="data/rankft/merged_hybrid_e5large_graph_candidates.jsonl"
GPU=$1
START=$2
STEP=$3

for i in $(seq $START $STEP 790); do
    CKPT="${DIR}/checkpoint-${i}"
    [ -f "${CKPT}/adapter_config.json" ] || continue
    
    for pool_info in "graph:${GRAPH_CAND}" "hybrid:${HYBRID_CAND}"; do
        POOL=$(echo $pool_info | cut -d: -f1)
        CAND=$(echo $pool_info | cut -d: -f2)
        OUT="${DIR}/eval_checkpoint-${i}_${POOL}"
        [ -f "${OUT}/summary.json" ] && continue
        CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
            --model_path ${MODEL} --lora_path ${CKPT} --test_data ${TEST} \
            --bm25_candidates ${CAND} --output_dir ${OUT} \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 4 2>/dev/null
    done
    
    G=$($PYTHON -c "import json; print(f\"{json.load(open('${DIR}/eval_checkpoint-${i}_graph/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    H=$($PYTHON -c "import json; print(f\"{json.load(open('${DIR}/eval_checkpoint-${i}_hybrid/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    echo "checkpoint-${i}: graph=${G}% hybrid=${H}%"
done
echo "DONE"
