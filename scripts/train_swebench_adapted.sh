#!/bin/bash
# Train SWE-bench-adapted reranker using combined GREPO + SWE-bench Full data.
#
# Strategy: Start from our best GREPO model (rankft_runB_graph/best)
# and continue fine-tuning on SWE-bench Full training data.
# This gives us domain adaptation without losing GREPO generalization.
#
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/train_swebench_adapted.sh

set -e

echo "=== SWE-bench Adapted Reranker Training ==="

# Check if training data exists
if [ ! -f "data/swebench_train/swebench_train.jsonl" ]; then
    echo "Error: Run scripts/prepare_swebench_train.py first"
    exit 1
fi

if [ ! -f "data/swebench_train/swebench_train_bm25_top500.jsonl" ]; then
    echo "Error: Run scripts/prepare_swebench_train.py first"
    exit 1
fi

# Continue from best GREPO model
python -u src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --train_data data/swebench_train/swebench_train.jsonl \
    --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
    --output_dir experiments/rankft_swebench_adapted \
    --device cuda:0 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_negatives 32 \
    --neg_bm25_ratio 0.75 \
    --neg_random_ratio 0.25 \
    --neg_graph_ratio 0.0 \
    --max_seq_length 512 \
    --save_steps 200 \
    2>&1 | tee logs/train_swebench_adapted.log

echo "Training complete!"
echo "Evaluate with:"
echo "  python src/eval/eval_rankft.py --lora_path experiments/rankft_swebench_adapted/best ..."
