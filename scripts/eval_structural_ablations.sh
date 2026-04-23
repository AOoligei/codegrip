#!/bin/bash
# Evaluate structural negative ablation models on GREPO
# Run after all 4 training runs complete
# Usage: bash scripts/eval_structural_ablations.sh GPU_ID
# e.g.:  bash scripts/eval_structural_ablations.sh 0

GPU_ID=${1:-0}
MODEL_PATH=/data/shuyang/models/Qwen2.5-7B-Instruct
CANDIDATES="data/rankft/merged_bm25_exp6_candidates.jsonl"
TEST_DATA="data/grepo_text/grepo_test.jsonl"

for ablation in rankft_ablation_samedir rankft_ablation_pathdist rankft_ablation_treeneighbor rankft_ablation_allstruct; do
    LORA_PATH="experiments/$ablation/best"
    OUTPUT_DIR="experiments/$ablation/eval_merged_rerank"

    if [ ! -f "$LORA_PATH/adapter_model.safetensors" ]; then
        echo "SKIP: $ablation/best not found (training not done?)"
        continue
    fi

    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo "SKIP: $ablation already evaluated"
        continue
    fi

    echo "=== Evaluating $ablation ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u src/eval/eval_rankft.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --bm25_candidates "$CANDIDATES" \
        --test_data "$TEST_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id 0 \
        --score_batch_size 16 \
        --max_seq_length 512 \
        --top_k 200
    echo
done

echo "=== Summary ==="
echo "Baseline (graph-hard neg):"
python3 -c "
import json
d = json.load(open('experiments/rankft_runB_graph/eval_merged_rerank/summary.json'))
o = d['overall']
print(f'  R@1={o[\"recall@1\"]:.2f}, R@5={o[\"recall@5\"]:.2f}, R@10={o[\"recall@10\"]:.2f}')
" 2>/dev/null || echo "  (no results)"

for ablation in rankft_ablation_samedir rankft_ablation_pathdist rankft_ablation_treeneighbor rankft_ablation_allstruct; do
    echo "$ablation:"
    python3 -c "
import json
d = json.load(open('experiments/$ablation/eval_merged_rerank/summary.json'))
o = d['overall']
print(f'  R@1={o[\"recall@1\"]:.2f}, R@5={o[\"recall@5\"]:.2f}, R@10={o[\"recall@10\"]:.2f}')
" 2>/dev/null || echo "  (no results yet)"
done
