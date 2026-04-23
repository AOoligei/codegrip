#!/bin/bash
# Wait for GPUs to free up, then launch all pending experiments
# Usage: nohup bash scripts/launch_when_gpu_free.sh > logs/launch_monitor.log 2>&1 &

set -uo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

wait_for_gpu() {
    local gpu=$1 min_free=$2
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
        if [ "$free" -ge "$min_free" ]; then
            echo "$(date): GPU $gpu free: ${free}MiB >= ${min_free}MiB"
            return 0
        fi
        sleep 60
    done
}

echo "=== Waiting for GPUs to free up ==="

# Phase 1: Code-residual v2 retraining (needs ~18G = 1 full GPU)
wait_for_gpu 0 20000
echo "$(date): Launching code-residual v2 on GPU 0"
nohup bash scripts/retrain_code_residual_v2.sh 0 > logs/retrain_v2.log 2>&1 &
RETRAIN_PID=$!

# Phase 2: SWE-bench perturbation evals (needs ~8G each)
# Wait for GPU 4 to free
wait_for_gpu 4 20000
echo "$(date): Launching SWE-bench perturbation evals"

# SWE-bench shuffle_filenames
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/swebench_adapted_5ep/checkpoint-210 \
    --test_data data/swebench_lite/swebench_perturb_shuffle_filenames_test.jsonl \
    --bm25_candidates data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl \
    --output_dir /data/chenlibin/grepo_agent_experiments/swebench_perturb/eval_shuffle_filenames \
    --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
    > logs/swebench_perturb_shuffle_filenames.log 2>&1 &

wait_for_gpu 5 20000
# SWE-bench shuffle_dirs
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/swebench_adapted_5ep/checkpoint-210 \
    --test_data data/swebench_lite/swebench_perturb_shuffle_dirs_test.jsonl \
    --bm25_candidates data/swebench_lite/swebench_perturb_shuffle_dirs_candidates.jsonl \
    --output_dir /data/chenlibin/grepo_agent_experiments/swebench_perturb/eval_shuffle_dirs \
    --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
    > logs/swebench_perturb_shuffle_dirs.log 2>&1 &

wait_for_gpu 6 20000
# SWE-bench flatten_dirs
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON scripts/eval_rankft_4bit.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/swebench_adapted_5ep/checkpoint-210 \
    --test_data data/swebench_lite/swebench_perturb_flatten_dirs_test.jsonl \
    --bm25_candidates data/swebench_lite/swebench_perturb_flatten_dirs_candidates.jsonl \
    --output_dir /data/chenlibin/grepo_agent_experiments/swebench_perturb/eval_flatten_dirs \
    --gpu_id 0 --top_k 50 --max_seq_length 1024 --score_batch_size 1 \
    > logs/swebench_perturb_flatten_dirs.log 2>&1 &

# Phase 3: 100-line ablation (after retrain v2 finishes, use its GPU)
wait $RETRAIN_PID
echo "$(date): Code-residual v2 done. Launching 100-line ablation on GPU 0"
nohup bash scripts/retrain_code_residual_100lines.sh 0 > logs/retrain_100lines.log 2>&1 &

echo "$(date): All experiments launched"
