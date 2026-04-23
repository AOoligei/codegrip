#!/bin/bash
# Evaluate negative mining ablation models
# Run after training completes

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
EVAL_CMD="$PYTHON src/eval/eval_rankft.py"
CANDIDATES="data/rankft/merged_bm25_exp6_candidates.jsonl"
TEST_DATA="data/grepo_text/grepo_test.jsonl"

for ablation in ablation_no_graph_neg ablation_random_neg; do
    LORA_PATH="experiments/$ablation/best"
    OUTPUT_DIR="experiments/$ablation/eval_merged_rerank"
    
    if [ ! -f "$LORA_PATH/adapter_model.safetensors" ]; then
        echo "SKIP: $ablation/best not found"
        continue
    fi
    
    echo "=== Evaluating $ablation ==="
    CUDA_VISIBLE_DEVICES=$1 $EVAL_CMD \
        --lora_path "$LORA_PATH" \
        --candidates "$CANDIDATES" \
        --test_data "$TEST_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_id 0 \
        --batch_size 16 \
        --max_seq_length 512
    echo
done

echo "=== Summary ==="
for ablation in ablation_no_graph_neg ablation_random_neg; do
    echo "$ablation:"
    cat "experiments/$ablation/eval_merged_rerank/summary.json" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
o = d.get('overall', d)
print(f'  R@1={o.get(\"recall@1\",o.get(\"hit@1\",\"?\")):.2f}, R@5={o.get(\"recall@5\",o.get(\"hit@5\",\"?\")):.2f}, R@10={o.get(\"recall@10\",o.get(\"hit@10\",\"?\")):.2f}')
" 2>/dev/null || echo "  (no results yet)"
done
