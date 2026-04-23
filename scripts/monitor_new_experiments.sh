#!/bin/bash
set -euo pipefail
# Monitor all 3 new training runs and auto-launch evals when done
# Checks every 5 min for training completion (final/ directory exists)

cd /home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
EVAL_SCRIPT=scripts/eval_rankft_4bit.py
MODEL_7B=/data/shuyang/models/Qwen2.5-7B-Instruct
MODEL_3B=/data/chenlibin/models/Qwen2.5-3B-Instruct
GRAPH_CAND=data/rankft/merged_bm25_exp6_candidates.jsonl
HYBRID_CAND=data/rankft/merged_hybrid_e5large_graph_candidates.jsonl
TEST_DATA=data/grepo_text/grepo_test.jsonl

echo "=== New Experiment Monitor Started $(date) ==="
echo "Watching: delex50_seed2 (GPU 2), scale_3B_delex50 (GPU 4), delex50_cfneg (GPU 7)"

SEED2_DONE=0
B3_DONE=0
COMBINED_DONE=0

while [ $SEED2_DONE -eq 0 ] || [ $B3_DONE -eq 0 ] || [ $COMBINED_DONE -eq 0 ]; do
    sleep 300

    # Check delex50_seed2
    if [ $SEED2_DONE -eq 0 ] && [ -d "experiments/rankft_delex50_seed2/final" ]; then
        echo "$(date): delex50_seed2 training DONE. Launching eval on GPU 2..."
        SEED2_DONE=1
        # Eval on graph and hybrid pools
        for pool_name in graph hybrid; do
            case $pool_name in
                graph)  CAND=$GRAPH_CAND ;;
                hybrid) CAND=$HYBRID_CAND ;;
            esac
            echo "  Eval $pool_name pool..."
            CUDA_VISIBLE_DEVICES=2 $PYTHON -u $EVAL_SCRIPT \
                --model_path $MODEL_7B \
                --lora_path experiments/rankft_delex50_seed2/final \
                --test_data $TEST_DATA \
                --bm25_candidates $CAND \
                --output_dir experiments/rankft_delex50_seed2/eval_${pool_name} \
                --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
            echo "  $pool_name done: $(date)"
        done
        echo "Seed2 eval complete."
    fi

    # Check scale_3B_delex50
    if [ $B3_DONE -eq 0 ] && [ -d "experiments/scale_3B_delex50/final" ]; then
        echo "$(date): 3B delex50 training DONE. Launching eval on GPU 4..."
        B3_DONE=1
        for pool_name in graph hybrid; do
            case $pool_name in
                graph)  CAND=$GRAPH_CAND ;;
                hybrid) CAND=$HYBRID_CAND ;;
            esac
            echo "  Eval $pool_name pool..."
            CUDA_VISIBLE_DEVICES=4 $PYTHON -u $EVAL_SCRIPT \
                --model_path $MODEL_3B \
                --lora_path experiments/scale_3B_delex50/final \
                --test_data $TEST_DATA \
                --bm25_candidates $CAND \
                --output_dir experiments/scale_3B_delex50/eval_${pool_name} \
                --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
            echo "  $pool_name done: $(date)"
        done
        echo "3B delex50 eval complete."
    fi

    # Check delex50_cfneg
    if [ $COMBINED_DONE -eq 0 ] && [ -d "experiments/rankft_delex50_cfneg/final" ]; then
        echo "$(date): delex50+cfneg training DONE. Launching eval on GPU 7..."
        COMBINED_DONE=1
        for pool_name in graph hybrid; do
            case $pool_name in
                graph)  CAND=$GRAPH_CAND ;;
                hybrid) CAND=$HYBRID_CAND ;;
            esac
            echo "  Eval $pool_name pool..."
            CUDA_VISIBLE_DEVICES=7 $PYTHON -u $EVAL_SCRIPT \
                --model_path $MODEL_7B \
                --lora_path experiments/rankft_delex50_cfneg/final \
                --test_data $TEST_DATA \
                --bm25_candidates $CAND \
                --output_dir experiments/rankft_delex50_cfneg/eval_${pool_name} \
                --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16
            echo "  $pool_name done: $(date)"
        done
        echo "Delex50+CF-Neg eval complete."
    fi

    echo "$(date): seed2=$SEED2_DONE 3B=$B3_DONE combined=$COMBINED_DONE"
done

echo ""
echo "=== ALL EXPERIMENTS COMPLETE $(date) ==="
echo "=== Results Summary ==="
for exp in rankft_delex50_seed2 scale_3B_delex50 rankft_delex50_cfneg; do
    echo "--- $exp ---"
    for pool in graph hybrid; do
        f="experiments/$exp/eval_${pool}/summary.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f'))['overall']; r1=d.get('recall@1',d.get('hit@1',0)); print(f'  $pool: R@1={r1:.2f}')" || echo "  $pool: R@1=N/A"
        fi
    done
done
