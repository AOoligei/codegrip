#!/bin/bash
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
AUTOMATION="${PYTHON} scripts/experiment_automation.py"
LOCK_DIR="${TMPDIR:-/tmp}/grepo_gpu_locks"
CURRENT_LOCK_PATH=""

cleanup_lock() {
    local lock_path="${1:-${CURRENT_LOCK_PATH:-}}"
    if [ -n "${lock_path}" ]; then
        ${AUTOMATION} release-lock --lock-path "${lock_path}" >/dev/null 2>&1 || true
        if [ "${lock_path}" = "${CURRENT_LOCK_PATH:-}" ]; then
            CURRENT_LOCK_PATH=""
        fi
    fi
}

trap 'cleanup_lock "${CURRENT_LOCK_PATH:-}"' EXIT INT TERM

claim_training_gpu() {
    local min_free_mb=$1
    local max_used_mb=$2
    local label=$3
    ${AUTOMATION} claim-gpu \
        --gpus 7 3 4 2 5 6 \
        --min-free-mb "${min_free_mb}" \
        --max-used-mb "${max_used_mb}" \
        --safety-buffer-mb 1536 \
        --lock-dir "${LOCK_DIR}" \
        --owner-pid "$$" \
        --label "${label}" 2>/dev/null
}

wait_for_training_gpu() {
    local description=$1
    local min_free_mb=$2
    local max_used_mb=$3
    local label=$4

    echo "[$(date)] Waiting for GPU for ${description}..."
    while true; do
        CLAIM=$(claim_training_gpu "${min_free_mb}" "${max_used_mb}" "${label}" || true)
        if [ -n "${CLAIM}" ]; then
            IFS=$'\t' read -r GPU CURRENT_LOCK_PATH <<< "${CLAIM}"
            return 0
        fi
        sleep 120
    done
}

resolve_adapter_dir() {
    ${AUTOMATION} resolve-adapter --exp-dir "$1" 2>/dev/null
}

run_training() {
    local gpu=$1
    shift
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} src/train/train_rankft.py "$@"
    local status=$?
    cleanup_lock
    return ${status}
}

# --- Experiment 1: 3B + delex50 ---
BASE_3B_LORA=$(resolve_adapter_dir "experiments/scale_3B_graph" || true)
if [ -z "${BASE_3B_LORA}" ]; then
    echo "[$(date)] Missing base adapter for 3B+delex50" >&2
    exit 1
fi
wait_for_training_gpu "3B+delex50" 12000 1000 "train-3b-delex50"
echo "[$(date)] Launching 3B+delex50 on GPU ${GPU}"
run_training "${GPU}" \
  --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
  --lora_path "${BASE_3B_LORA}" \
  --train_data data/swebench_train/swebench_train.jsonl \
  --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
  --file_tree_dir data/swebench_lite/file_trees \
  --output_dir experiments/swebench_adapted_3B_delex50 \
  --device cuda:0 \
  --num_negatives 16 \
  --neg_bm25_ratio 0.75 \
  --neg_graph_ratio 0.0 \
  --neg_random_ratio 0.25 \
  --learning_rate 1e-5 \
  --num_epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 1024 \
  --lora_rank 32 \
  --seed 42 \
  --delex_fraction 0.5
echo "[$(date)] 3B+delex50 training done"

# --- Experiment 2: 5ep + lr=2e-5 ---
BASE_7B_LORA=$(resolve_adapter_dir "experiments/rankft_runB_graph" || true)
if [ -z "${BASE_7B_LORA}" ]; then
    echo "[$(date)] Missing base adapter for 5ep+lr2e5" >&2
    exit 1
fi
wait_for_training_gpu "5ep+lr2e5" 21000 512 "train-5ep-lr2e5"
echo "[$(date)] Launching 5ep+lr2e5 on GPU ${GPU}"
run_training "${GPU}" \
  --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
  --lora_path "${BASE_7B_LORA}" \
  --train_data data/swebench_train/swebench_train.jsonl \
  --bm25_candidates data/swebench_train/swebench_train_bm25_top500.jsonl \
  --file_tree_dir data/swebench_lite/file_trees \
  --output_dir experiments/swebench_adapted_5ep_lr2e5 \
  --device cuda:0 \
  --num_negatives 16 \
  --neg_bm25_ratio 0.75 \
  --neg_graph_ratio 0.0 \
  --neg_random_ratio 0.25 \
  --learning_rate 2e-5 \
  --num_epochs 5 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 1024 \
  --lora_rank 32 \
  --seed 42 \
  --delex_fraction 0.0
echo "[$(date)] 5ep+lr2e5 training done"
