#!/bin/bash
# Run PathSwap eval for Llama-3.1-8B and Qwen3-8B on GREPO
# Usage: bash scripts/eval_pathswap_cross_llm.sh <GPU_ID>
set -e
GPU=${1:-7}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

run_one () {
    local NAME=$1 MODEL=$2 LORA=$3
    echo "=== PathSwap eval: $NAME on GPU $GPU === $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_rankft_4bit.py \
        --model_path "$MODEL" \
        --lora_path "$LORA" \
        --test_data data/pathswap/grepo_test_pathswap.jsonl \
        --bm25_candidates data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl \
        --output_dir experiments/pathswap_eval/${NAME}_graph \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 8
    local F=experiments/pathswap_eval/${NAME}_graph/summary.json
    if [ -f "$F" ]; then
        $PY -c "import json; d=json.load(open('$F'))['overall']; print(f'  $NAME PathSwap R@1={d.get(\"recall@1\",d.get(\"hit@1\",0)):.2f} R@5={d.get(\"recall@5\",d.get(\"hit@5\",0)):.2f}')"
    fi
}

run_one llama31_8b /data/hzy/models/Llama-3.1-8B-Instruct experiments/cross_llm_llama31_8b/best
run_one qwen3_8b   /data/hzy/models/Qwen3-8B           experiments/cross_llm_qwen3_8b/best

echo "=== DONE === $(date)"
