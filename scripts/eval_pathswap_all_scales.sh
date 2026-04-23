#!/bin/bash
# Run PathSwap-GREPO eval across all scale models
# Evaluates on graph pool only (main comparison)
# Uses GPUs 5, 6, 7 in parallel

PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
TEST_DATA=data/pathswap/grepo_test_pathswap.jsonl
GRAPH_CAND=data/pathswap/merged_bm25_exp6_candidates_pathswap.jsonl
HYBRID_CAND=data/pathswap/merged_hybrid_e5large_graph_candidates_pathswap.jsonl

mkdir -p experiments/pathswap_eval

eval_model() {
    local GPU=$1
    local MODEL=$2
    local LORA=$3
    local NAME=$4

    echo "[$NAME] Starting on GPU $GPU at $(date)"

    # Graph pool
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA \
        --test_data $TEST_DATA \
        --bm25_candidates $GRAPH_CAND \
        --output_dir experiments/pathswap_eval/${NAME}_graph \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

    # Hybrid pool
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA \
        --test_data $TEST_DATA \
        --bm25_candidates $HYBRID_CAND \
        --output_dir experiments/pathswap_eval/${NAME}_hybrid \
        --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16

    echo "[$NAME] Done at $(date)"
    for pool in graph hybrid; do
        f="experiments/pathswap_eval/${NAME}_${pool}/summary.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'  ${NAME}_${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
        fi
    done
}

# Launch in parallel on different GPUs
# 0.5B and 1.5B are small, can share a GPU sequentially
# 3B and 7B each get their own GPU

# GPU 7: 7B (largest, needs most memory)
eval_model 7 /data/shuyang/models/Qwen2.5-7B-Instruct experiments/rankft_runB_graph/best 7B &
PID_7B=$!

# GPU 5: 0.5B then 1.5B (sequential, small models)
(eval_model 5 /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct experiments/scale_0.5B_graph/final 0.5B && \
 eval_model 5 /data/chenlibin/models/Qwen2.5-1.5B-Instruct experiments/scale_1.5B_graph/final 1.5B) &
PID_SMALL=$!

# GPU 6: 3B
eval_model 6 /data/chenlibin/models/Qwen2.5-3B-Instruct experiments/scale_3B_graph/final 3B &
PID_3B=$!

echo "Launched: 7B (GPU 7, PID $PID_7B), 0.5B+1.5B (GPU 5, PID $PID_SMALL), 3B (GPU 6, PID $PID_3B)"
echo "Waiting for all to complete..."

wait $PID_7B $PID_SMALL $PID_3B

echo ""
echo "========================================="
echo "PathSwap-GREPO Results Summary"
echo "========================================="
for name in 0.5B 1.5B 3B 7B; do
    for pool in graph hybrid; do
        f="experiments/pathswap_eval/${name}_${pool}/summary.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f'))['overall']; print(f'${name}_${pool}: R@1={d.get(\"recall@1\", d.get(\"hit@1\",0)):.2f}, R@5={d.get(\"recall@5\", d.get(\"hit@5\",0)):.2f}')"
        fi
    done
done
echo "Done: $(date)"
