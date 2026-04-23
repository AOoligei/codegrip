#!/bin/bash
# Eval CF-Neg B reranker on graph, hybrid, BM25 pools
GPU_ID=${1:-5}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
TEST_DATA=data/grepo_text/grepo_test.jsonl
MODEL_PATH=/data/shuyang/models/Qwen2.5-7B-Instruct
LORA_PATH=experiments/rankft_cfneg_B/final

echo "=== CF-Neg B Eval (GPU $GPU_ID) ==="
echo "Start: $(date)"

if [ ! -d "$LORA_PATH" ]; then
    LORA_PATH=experiments/rankft_cfneg_B/best
    echo "Warning: using 'best' checkpoint"
fi

for pool_name in graph hybrid bm25; do
    case $pool_name in
        graph)  CAND=data/rankft/merged_bm25_exp6_candidates.jsonl ;;
        hybrid) CAND=data/rankft/merged_hybrid_e5large_graph_candidates.jsonl ;;
        bm25)   CAND=data/rankft/grepo_test_bm25_top500.jsonl ;;
    esac
    echo ""
    echo ">>> [$pool_name pool] $(date)"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
        --model_path $MODEL_PATH \
        --lora_path $LORA_PATH \
        --test_data $TEST_DATA \
        --bm25_candidates $CAND \
        --output_dir experiments/rankft_cfneg_B/eval_${pool_name} \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
    echo "  $pool_name done: $(date)"
done

echo ""
echo "=== Summary ==="
for pool in graph hybrid bm25; do
    f="experiments/rankft_cfneg_B/eval_${pool}/summary.json"
    if [ -f "$f" ]; then
        $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
    fi
done
echo "End: $(date)"
