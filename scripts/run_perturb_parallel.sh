#!/bin/bash
# Run 4 remaining perturbation conditions in PARALLEL on 4 free GPUs
# Each condition on its own GPU to avoid OOM

cd /home/chenlibin/grepo_agent

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/rankft_runB_graph/best

echo "=== Parallel Perturbation Eval (4bit) ==="
echo "Start: $(date)"

run_one() {
    local cond=$1
    local gpu=$2
    local DIR="experiments/path_perturb_${cond}"
    local OUT="$DIR/eval_4bit"

    if [ -f "$OUT/summary.json" ]; then
        echo "[$cond] SKIP: already done"
        return
    fi

    echo "[$cond] Starting on GPU $gpu at $(date)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u scripts/eval_rankft_4bit.py \
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
        echo "[$cond] DONE: R@1=$R1"
    else
        echo "[$cond] FAILED"
    fi
}

# Launch all 4 in parallel, each on its own GPU
run_one shuffle_filenames 0 &
run_one remove_module_names 2 &
run_one flatten_dirs 3 &
run_one swap_leaf_dirs 4 &

# Wait for all
wait

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
