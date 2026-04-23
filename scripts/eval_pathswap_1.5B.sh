#!/bin/bash
# Re-run PathSwap eval for 1.5B on a free GPU (1.5B OOMed on shared GPU 5)
GPU=${1:-3}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

TEST_DATA=data/pathswap/grepo_test_pathswap.jsonl
GRAPH_CAND=data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl
HYBRID_CAND=data/pathswap/merged_hybrid_e5large_graph_candidates_pathswap.jsonl
MODEL=/data/chenlibin/models/Qwen2.5-1.5B-Instruct
LORA=experiments/scale_1.5B_graph/final

echo "[1.5B] Starting PathSwap eval on GPU $GPU at $(date)"

# Graph pool
echo ">>> Graph pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data $TEST_DATA \
    --bm25_candidates $GRAPH_CAND \
    --output_dir experiments/pathswap_eval/1.5B_graph \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

# Hybrid pool
echo ">>> Hybrid pool"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/eval_rankft_4bit.py \
    --model_path $MODEL \
    --lora_path $LORA \
    --test_data $TEST_DATA \
    --bm25_candidates $HYBRID_CAND \
    --output_dir experiments/pathswap_eval/1.5B_hybrid \
    --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

echo "[1.5B] Done at $(date)"
for pool in graph hybrid; do
    f="experiments/pathswap_eval/1.5B_${pool}/summary.json"
    if [ -f "$f" ]; then
        $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'1.5B_{pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
    else
        echo "1.5B_${pool}: FAILED (no summary)"
    fi
done
