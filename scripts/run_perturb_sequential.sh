#!/bin/bash
# Run remaining perturbation conditions ONE AT A TIME on GPU 4
# Each takes ~45-60 min with 4-bit, batch=16

GPU_ID=4
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/rankft_runB_graph/best

cd /home/chenlibin/grepo_agent

CONDITIONS="shuffle_filenames remove_module_names flatten_dirs swap_leaf_dirs"

echo "=== Sequential Perturbation Eval (4bit, GPU $GPU_ID) ==="
echo "Start: $(date)"

for cond in $CONDITIONS; do
    DIR="experiments/path_perturb_${cond}"
    OUT="$DIR/eval_4bit"

    if [ -f "$OUT/summary.json" ]; then
        echo "SKIP $cond: already done"
        continue
    fi

    if [ ! -f "$DIR/test.jsonl" ]; then
        echo "SKIP $cond: no data"
        continue
    fi

    echo ""
    echo ">>> [$cond] $(date)"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA \
        --test_data "$DIR/test.jsonl" \
        --bm25_candidates "$DIR/bm25_candidates.jsonl" \
        --output_dir "$OUT" \
        --gpu_id 0 \
        --top_k 200 \
        --max_seq_length 512 \
        --score_batch_size 16

    if [ -f "$OUT/summary.json" ]; then
        R1=$($PYTHON -c "import json; d=json.load(open('$OUT/summary.json'))['overall']; print(f'{d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}')")
        echo "  DONE $cond: R@1=$R1"
    else
        echo "  FAILED $cond"
    fi
done

echo ""
echo "=== All Done ==="
echo "End: $(date)"
echo ""
echo "=== Final Summary ==="
printf "%-25s %8s %8s\n" "Condition" "R@1" "R@5"
echo "------------------------------------------------"
printf "%-25s %8s %8s\n" "Normal (baseline)" "27.01" "49.17"
for cond in shuffle_dirs shuffle_filenames remove_module_names flatten_dirs swap_leaf_dirs; do
    OUT="experiments/path_perturb_${cond}/eval_4bit/summary.json"
    if [ -f "$OUT" ]; then
        R1=$($PYTHON -c "import json; d=json.load(open('$OUT'))['overall']; print(f'{d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}')")
        R5=$($PYTHON -c "import json; d=json.load(open('$OUT'))['overall']; print(f'{d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')")
        printf "%-25s %8s %8s\n" "$cond" "$R1" "$R5"
    fi
done
