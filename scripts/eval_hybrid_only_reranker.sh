#!/bin/bash
# Eval hybrid-only-trained reranker on all 3 pools: graph, hybrid, BM25
GPU_ID=${1:-2}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
TEST_DATA=data/grepo_text/grepo_test.jsonl
MODEL_PATH=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA_PATH=experiments/rankft_hybrid_only_v2/final

echo "=== Hybrid-Only Reranker Eval (GPU $GPU_ID) ==="
echo "Start: $(date)"

if [ ! -d "$LORA_PATH" ]; then
    LORA_PATH=experiments/rankft_hybrid_only_v2/best
    echo "Warning: using 'best' checkpoint (training may not be done)"
fi

# Graph pool
echo ""
echo ">>> [Graph pool] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data $TEST_DATA \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/rankft_hybrid_only_v2/eval_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
echo "  Graph done: $(date)"

# Hybrid pool
echo ""
echo ">>> [Hybrid pool] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data $TEST_DATA \
    --bm25_candidates data/rankft/merged_hybrid_e5large_graph_candidates.jsonl \
    --output_dir experiments/rankft_hybrid_only_v2/eval_hybrid \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
echo "  Hybrid done: $(date)"

# BM25-only pool
echo ""
echo ">>> [BM25 pool] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL_PATH \
    --lora_path $LORA_PATH \
    --test_data $TEST_DATA \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir experiments/rankft_hybrid_only_v2/eval_bm25 \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
echo "  BM25 done: $(date)"

echo ""
echo "=== Summary ==="
for pool in graph hybrid bm25; do
    f="experiments/rankft_hybrid_only_v2/eval_${pool}/summary.json"
    if [ -f "$f" ]; then
        $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
    fi
done
echo "End: $(date)"
