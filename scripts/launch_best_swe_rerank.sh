#!/bin/bash
# Launch reranking on best BM25 ensemble for SWE-bench
# Run on GPU 6 after swebench_path reranking finishes

echo "Waiting for GPU 6 SWE-bench path reranking to finish (PID 2135550)..."
while kill -0 2135550 2>/dev/null; do
    sleep 30
done
echo "GPU 6 is free! Launching best ensemble reranking..."

CUDA_VISIBLE_DEVICES=6 python -u src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_final_top500.jsonl \
    --top_k 500 \
    --max_seq_length 512 \
    --score_batch_size 16 \
    --gpu_id 0 \
    --output_dir experiments/rankft_runB_graph/eval_swebench_best_ensemble \
    2>&1 | tee logs/swebench_best_ensemble_rerank.log

echo "Best ensemble reranking complete!"
