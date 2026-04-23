#!/bin/bash
# Run path perturbation evaluation for all conditions.
# Usage: bash scripts/run_path_perturbation_eval.sh <GPU_ID>
#
# Prerequisites: run the data creation step first:
#   python scripts/controlled_path_perturbation.py

set -euo pipefail

GPU_ID=${1:?Usage: $0 <GPU_ID>}
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_PATH="experiments/rankft_runB_graph/best"

CONDITIONS=(
    shuffle_dirs
    shuffle_filenames
    remove_module_names
    flatten_dirs
    swap_leaf_dirs
)

cd "$BASE_DIR"

echo "=========================================="
echo "Path Perturbation Evaluation"
echo "GPU: $GPU_ID  Model: $MODEL_PATH"
echo "LoRA: $LORA_PATH"
echo "=========================================="

for COND in "${CONDITIONS[@]}"; do
    EXP_DIR="experiments/path_perturb_${COND}"
    TEST_FILE="${EXP_DIR}/test.jsonl"
    BM25_FILE="${EXP_DIR}/bm25_candidates.jsonl"
    OUT_DIR="${EXP_DIR}/eval"

    if [ ! -f "$TEST_FILE" ]; then
        echo "[SKIP] $COND: $TEST_FILE not found"
        continue
    fi

    echo ""
    echo "=== Evaluating: $COND ==="
    $PYTHON src/eval/eval_rankft.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --test_data "$TEST_FILE" \
        --bm25_candidates "$BM25_FILE" \
        --output_dir "$OUT_DIR" \
        --gpu_id "$GPU_ID" \
        --top_k 200 \
        --max_seq_length 512
done

# Print summary table
echo ""
echo "=========================================="
echo "Path Perturbation Results Summary"
echo "=========================================="
printf "%-25s %8s %8s %8s\n" "Condition" "R@1" "R@5" "R@10"
printf "%-25s %8s %8s %8s\n" "-------------------------" "--------" "--------" "--------"
printf "%-25s %8s %8s %8s\n" "Normal (baseline)" "24.50" "-" "-"

for COND in "${CONDITIONS[@]}"; do
    SUMMARY="experiments/path_perturb_${COND}/eval/summary.json"
    if [ -f "$SUMMARY" ]; then
        R1=$($PYTHON -c "
import json
d = json.load(open('$SUMMARY'))['overall']
print(f\"{d.get('recall@1', d.get('hit@1', 0)):.2f}\")
")
        R5=$($PYTHON -c "
import json
d = json.load(open('$SUMMARY'))['overall']
print(f\"{d.get('recall@5', d.get('hit@5', 0)):.2f}\")
")
        R10=$($PYTHON -c "
import json
d = json.load(open('$SUMMARY'))['overall']
print(f\"{d.get('recall@10', d.get('hit@10', 0)):.2f}\")
")
        printf "%-25s %8s %8s %8s\n" "$COND" "$R1" "$R5" "$R10"
    else
        printf "%-25s %8s %8s %8s\n" "$COND" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "Interpretation guide:"
echo "  shuffle_dirs:        If R@1 drops -> model uses directory=structural proximity"
echo "  shuffle_filenames:   If R@1 drops -> model uses filename semantics"
echo "  remove_module_names: If R@1 drops -> model uses module naming conventions"
echo "  flatten_dirs:        If R@1 drops -> directory hierarchy matters beyond filenames"
echo "  swap_leaf_dirs:      If R@1 drops -> fine-grained directory locality matters"
