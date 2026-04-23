#!/bin/bash
echo "Waiting for GPU 0 function-BM25 reranking to finish (PID 2188700)..."
while kill -0 2188700 2>/dev/null; do
    sleep 30
done
echo "GPU 0 free! Launching chunk BM25 reranking..."

CUDA_VISIBLE_DEVICES=0 python -u src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_bm25_chunk_top500.jsonl \
    --top_k 500 \
    --max_seq_length 512 \
    --score_batch_size 16 \
    --gpu_id 0 \
    --output_dir experiments/rankft_runB_graph/eval_swebench_chunk \
    2>&1 | tee logs/swebench_chunk_rerank.log

echo "Chunk BM25 reranking complete!"
