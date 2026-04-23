#!/bin/bash
# Eval random-masking control on graph-expanded pool
# Usage: bash scripts/eval_random_mask.sh [GPU_ID]

GPU_ID=${1:-3}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/rankft_runB_graph/best
EVAL_SCRIPT=scripts/eval_rankft_4bit.py

cd /home/chenlibin/grepo_agent

DIR="experiments/path_perturb_random_mask"
OUT="$DIR/eval_4bit"

echo "=== Random-Mask Control Eval (4-bit, GPU $GPU_ID) ==="
echo "Control for delexicalization: masks same NUMBER of path tokens"
echo "but chosen RANDOMLY, not by issue-text overlap."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $EVAL_SCRIPT \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data "$DIR/test.jsonl" \
    --bm25_candidates "$DIR/bm25_candidates.jsonl" \
    --output_dir "$OUT" \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    --score_batch_size 16

echo ""
echo "=== Result ==="
if [ -f "$OUT/summary.json" ]; then
    $PYTHON -c "
import json
d = json.load(open('$OUT/summary.json'))['overall']
r1 = d.get('recall@1', d.get('hit@1', 0))
r5 = d.get('recall@5', d.get('hit@5', 0))
print(f'Random-mask R@1={r1:.2f}% R@5={r5:.2f}%')
print(f'Compare: Delex R@1=8.85% (issue-overlap masking)')
print(f'Compare: Baseline R@1=27.01% (no masking)')
print()
if r1 > 20:
    print('CONCLUSION: Random masking does NOT cause collapse.')
    print('The delex collapse is specific to issue-overlap tokens (shortcut).')
elif r1 < 12:
    print('CONCLUSION: Random masking ALSO causes collapse.')
    print('The effect may be generic noise sensitivity, not shortcut-specific.')
else:
    print('CONCLUSION: Partial collapse. Mixed evidence.')
"
fi
