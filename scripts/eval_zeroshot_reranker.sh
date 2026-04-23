#!/bin/bash
# Zero-shot reranker evaluation: test if base/SFT model can rerank BM25 candidates
# without any RankFT training. Uses the Yes/No prompt.
GPU_ID=${1:-5}
LORA=${2:-""}  # optional: experiments/exp1_sft_only/stage2_sft/final

cd /home/chenlibin/grepo_agent

LORA_ARG=""
LABEL="zeroshot_base"
if [ -n "$LORA" ]; then
    LORA_ARG="--lora_path $LORA"
    LABEL="zeroshot_sft"
fi

echo "=== Zero-shot reranker eval (${LABEL}) on GREPO ==="
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 \
/home/chenlibin/miniconda3/envs/tgn/bin/python src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    $LORA_ARG \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir experiments/${LABEL}_grepo \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    --score_batch_size 16

echo ""
echo "=== Zero-shot reranker eval (${LABEL}) on SWE-bench ==="
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 \
/home/chenlibin/miniconda3/envs/tgn/bin/python src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    $LORA_ARG \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir experiments/${LABEL}_swebench \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    --score_batch_size 16
