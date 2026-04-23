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
START=$2  # start from this checkpoint number
STEP=$3   # step between checkpoints to eval

for i in $(seq $START $STEP 790); do
    CKPT="${DIR}/checkpoint-${i}"
    [ -f "${CKPT}/adapter_config.json" ] || continue
    
    # Graph
    OUTG="${DIR}/eval_checkpoint-${i}_graph"
    if [ ! -f "${OUTG}/summary.json" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
            --model_path ${MODEL} --lora_path ${CKPT} --test_data ${TEST} \
            --bm25_candidates ${GRAPH_CAND} --output_dir ${OUTG} \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 1 2>/dev/null
    fi
    
    # Hybrid
    OUTH="${DIR}/eval_checkpoint-${i}_hybrid"
    if [ ! -f "${OUTH}/summary.json" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} $PYTHON scripts/eval_rankft_4bit.py \
            --model_path ${MODEL} --lora_path ${CKPT} --test_data ${TEST} \
            --bm25_candidates ${HYBRID_CAND} --output_dir ${OUTH} \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 1 2>/dev/null
    fi
    
    # Print paired result
    G=$($PYTHON -c "import json; print(f\"{json.load(open('${OUTG}/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    H=$($PYTHON -c "import json; print(f\"{json.load(open('${OUTH}/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    echo "checkpoint-${i}: graph=${G}% hybrid=${H}%"
done
