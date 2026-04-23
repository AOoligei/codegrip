#!/bin/bash
set -euo pipefail
# Wait for delex50 training to finish, then eval on the freed GPU 3
cd /home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3

echo "Waiting for delex50 training to complete..."
while true; do
    if [ -d "experiments/rankft_delex50/final" ]; then
        echo "Delex50 training done at $(date)"
        sleep 30

        LORA=experiments/rankft_delex50/final
        if [ ! -d "$LORA" ]; then
            LORA=experiments/rankft_delex50/best
        fi

        echo "Launching delex50 3-pool eval on GPU 3..."

        # Graph pool
        echo ">>> Graph pool $(date)"
        CUDA_VISIBLE_DEVICES=3 $PYTHON -u scripts/eval_rankft_4bit.py \
            --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
            --lora_path $LORA \
            --test_data data/grepo_text/grepo_test.jsonl \
            --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
            --output_dir experiments/rankft_delex50/eval_graph \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

        # Hybrid pool
        echo ">>> Hybrid pool $(date)"
        CUDA_VISIBLE_DEVICES=3 $PYTHON -u scripts/eval_rankft_4bit.py \
            --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
            --lora_path $LORA \
            --test_data data/grepo_text/grepo_test.jsonl \
            --bm25_candidates data/rankft/merged_hybrid_e5large_graph_candidates.jsonl \
            --output_dir experiments/rankft_delex50/eval_hybrid \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

        # Summary
        echo ""
        echo "=== Delex50 Eval Summary ==="
        for pool in graph hybrid; do
            f="experiments/rankft_delex50/eval_${pool}/summary.json"
            if [ -f "$f" ]; then
                $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')" || echo "${pool}: R@1=N/A, R@5=N/A"
            fi
        done
        echo "End: $(date)"
        exit 0
    fi
    sleep 120
done
