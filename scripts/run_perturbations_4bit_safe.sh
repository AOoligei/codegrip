#!/bin/bash
# Run all path perturbation evals with 4-bit quantization (safe for shared GPUs)
# Usage: CUDA_VISIBLE_DEVICES=6 bash scripts/run_perturbations_4bit_safe.sh

set -e
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/rankft_runB_graph/best
GPU_ID=${1:-6}

cd /home/chenlibin/grepo_agent

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDITIONS="shuffle_dirs shuffle_filenames remove_module_names flatten_dirs swap_leaf_dirs"

echo "=== Path Perturbation Eval (4-bit, GPU $GPU_ID) ==="
echo "Start: $(date)"

for cond in $CONDITIONS; do
    DIR="experiments/path_perturb_${cond}"
    OUT="$DIR/eval_4bit"

    if [ ! -f "$DIR/test.jsonl" ]; then
        echo "SKIP $cond: no test data"
        continue
    fi

    if [ -f "$OUT/summary.json" ]; then
        echo "SKIP $cond: already done"
        continue
    fi

    echo ""
    echo ">>> [$cond] $(date)"
    $PYTHON scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA \
        --test_data "$DIR/test.jsonl" \
        --bm25_candidates "$DIR/bm25_candidates.jsonl" \
        --output_dir "$OUT" \
        --gpu_id $GPU_ID \
        --top_k 200 \
        --max_seq_length 512 \
        --score_batch_size 4

    if [ -f "$OUT/summary.json" ]; then
        $PYTHON -c "import json; d=json.load(open('$OUT/summary.json'))['overall']; print(f'  $cond: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
    else
        echo "  $cond: FAILED (no summary)"
    fi
done

echo ""
echo "=== All Results ==="
printf "%-25s %8s %8s\n" "Condition" "R@1" "R@5"
echo "------------------------------------------------"
for cond in $CONDITIONS; do
    OUT="experiments/path_perturb_${cond}/eval_4bit/summary.json"
    if [ -f "$OUT" ]; then
        $PYTHON -c "import json; d=json.load(open('$OUT'))['overall']; print(f'{\"$cond\":<25s} {d.get(\"recall@1\", d.get(\"hit@1\",0)):>8.2f} {d.get(\"recall@5\", d.get(\"hit@5\",0)):>8.2f}')"
    fi
done
echo "=== Done: $(date) ==="
