#!/bin/bash
# Evaluate SWE-bench reranker on BM25 and hybrid pools to test oracle fallacy
# Usage: bash scripts/swebench_eval_oracle_fallacy.sh [GPU_ID]

GPU_ID=${1:-3}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA=experiments/swebench_rankft_bm25/final
EVAL_SCRIPT=scripts/eval_rankft_4bit.py

cd /home/chenlibin/grepo_agent

echo "=== SWE-bench Oracle Fallacy Eval (4-bit, GPU $GPU_ID) ==="
echo ""

# Eval on BM25 pool (native - trained on this distribution)
echo "--- Eval on BM25 pool (native) ---"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u $EVAL_SCRIPT \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir experiments/swebench_rankft_bm25/eval_bm25 \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    --score_batch_size 4

echo ""
echo "--- Eval on Hybrid pool (shifted, higher oracle recall) ---"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u $EVAL_SCRIPT \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_hybrid_bm25_e5large.jsonl \
    --output_dir experiments/swebench_rankft_bm25/eval_hybrid \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    --score_batch_size 4

echo ""
echo "=== Results Comparison ==="
$PYTHON -c "
import json

pools = [
    ('BM25 (native)', 'experiments/swebench_rankft_bm25/eval_bm25/summary.json'),
    ('Hybrid (shifted)', 'experiments/swebench_rankft_bm25/eval_hybrid/summary.json'),
]
for name, path in pools:
    try:
        d = json.load(open(path))['overall']
        r1 = d.get('recall@1', d.get('hit@1', 0))
        r5 = d.get('recall@5', d.get('hit@5', 0))
        print(f'{name:20s}: R@1={r1:.2f}% R@5={r5:.2f}%')
    except FileNotFoundError:
        print(f'{name:20s}: [not yet available]')

print()
print('Oracle recall: BM25=86.67%, Hybrid=91.67% (+5.0pp)')
print('If R@1 drops on hybrid despite higher oracle → oracle fallacy confirmed on SWE-bench')
"
