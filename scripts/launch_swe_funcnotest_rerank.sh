#!/bin/bash
# Launch reranking on function-notest BM25 for SWE-bench
# Run on GPU 7 after swebench_content reranking finishes

echo "Waiting for GPU 7 SWE-bench content reranking to finish (PID 2165480)..."
while kill -0 2165480 2>/dev/null; do
    sleep 30
done
echo "GPU 7 is free! Launching function-notest reranking..."

CUDA_VISIBLE_DEVICES=7 python -u src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_function_notest_top500.jsonl \
    --top_k 500 \
    --max_seq_length 512 \
    --score_batch_size 16 \
    --gpu_id 0 \
    --output_dir experiments/rankft_runB_graph/eval_swebench_funcnotest \
    2>&1 | tee logs/swebench_funcnotest_rerank.log

echo "Function-notest reranking complete!"
