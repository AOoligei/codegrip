#!/bin/bash
cd ~/grepo_agent
P=~/miniconda3/envs/tgn/bin/python3
MODEL=~/models/Llama-3.1-8B-Instruct
LORA=experiments/cross_llm_llama31_8b/best
GPU=$1
shift
CONDS="$@"

for cond in $CONDS; do
    if [ "$cond" = "anonymized" ]; then DIR=experiments/path_anonymized; else DIR=experiments/path_perturb_${cond}; fi
    OUT=experiments/cross_llm_llama31_8b/eval_perturb_${cond}
    [ -f "${DIR}/test.jsonl" ] || { echo "SKIP $cond: no data"; continue; }
    [ -f "${OUT}/summary.json" ] && { echo "SKIP $cond: done"; continue; }
    echo "Running ${cond} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} $P scripts/eval_rankft_4bit.py --model_path $MODEL --lora_path $LORA \
        --test_data ${DIR}/test.jsonl --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir $OUT --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 4 2>/dev/null
    H=$($P -c "import json; print(f\"{json.load(open('${OUT}/summary.json'))['overall']['hit@1']:.2f}\")" 2>/dev/null || echo "?")
    echo "  ${cond}: ${H}%"
done
echo "DONE"
