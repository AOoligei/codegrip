#!/bin/bash
# Run all path perturbation evals with 4-bit quantization (fits in ~6GB, batch=16 works)
# Usage: bash scripts/run_all_perturbations_4bit.sh [GPU_ID]

GPU_ID=${1:-4}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/rankft_runB_graph/best
EVAL_SCRIPT=scripts/eval_rankft_4bit.py

cd /home/chenlibin/grepo_agent

CONDITIONS="shuffle_dirs shuffle_filenames remove_module_names flatten_dirs swap_leaf_dirs"

echo "=== Path Perturbation Eval (4-bit, GPU $GPU_ID) ==="
echo "Start: $(date)"
echo ""

for cond in $CONDITIONS; do
    DIR="experiments/path_perturb_${cond}"
    OUT="$DIR/eval_4bit"

    if [ ! -f "$DIR/test.jsonl" ]; then
        echo "SKIP $cond: no data"
        continue
    fi

    if [ -f "$OUT/summary.json" ]; then
        R1=$($PYTHON -c "import json; d=json.load(open('$OUT/summary.json'))['overall']; print(f'{d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}')")
        echo "SKIP $cond: already done (R@1=$R1)"
        continue
    fi

    echo ">>> [$cond] $(date)"
    $PYTHON $EVAL_SCRIPT \
        --model_path $MODEL \
        --lora_path $LORA \
        --test_data "$DIR/test.jsonl" \
        --bm25_candidates "$DIR/bm25_candidates.jsonl" \
        --output_dir "$OUT" \
        --gpu_id $GPU_ID \
        --top_k 200 \
        --max_seq_length 512 \
        --score_batch_size 16

    if [ -f "$OUT/summary.json" ]; then
        R1=$($PYTHON -c "import json; d=json.load(open('$OUT/summary.json'))['overall']; print(f'{d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}')")
        R5=$($PYTHON -c "import json; d=json.load(open('$OUT/summary.json'))['overall']; print(f'{d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')")
        echo "  $cond: R@1=$R1, R@5=$R5"
    fi
    echo ""
done

echo ""
echo "=== Summary ==="
echo "End: $(date)"
printf "%-25s %8s %8s\n" "Condition" "R@1" "R@5"
echo "------------------------------------------------"
printf "%-25s %8s %8s\n" "Normal (baseline)" "27.01" "49.17"
for cond in $CONDITIONS; do
    OUT="experiments/path_perturb_${cond}/eval_4bit/summary.json"
    if [ -f "$OUT" ]; then
        R1=$($PYTHON -c "import json; d=json.load(open('$OUT'))['overall']; print(f'{d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}')")
        R5=$($PYTHON -c "import json; d=json.load(open('$OUT'))['overall']; print(f'{d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')")
        printf "%-25s %8s %8s\n" "$cond" "$R1" "$R5"
    fi
done
