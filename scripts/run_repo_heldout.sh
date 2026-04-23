#!/bin/bash
# Repo-held-out retraining + evaluation for CodeGRIP.
#
# Tests whether path dependency holds on completely unseen repos.
# Usage: bash scripts/run_repo_heldout.sh GPU_ID
#        nohup bash scripts/run_repo_heldout.sh 3 > logs/repo_heldout.log 2>&1 &

set -euo pipefail

GPU_ID=${1:?Usage: bash scripts/run_repo_heldout.sh GPU_ID}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

MODEL=/data/shuyang/models/Qwen2.5-7B-Instruct
DATA_DIR=/data/chenlibin/grepo_agent_experiments/repo_heldout
EXP_DIR=/data/chenlibin/grepo_agent_experiments/repo_heldout

echo "=============================================="
echo "Repo-held-out experiment (GPU $GPU_ID)"
echo "Start: $(date)"
echo "=============================================="

# ---- Step 0: Prepare data split ----
echo ""
echo "=== Step 0: Prepare repo-held-out data split ==="
$PYTHON scripts/prepare_repo_heldout.py

# ---- Step 1: Train reranker on filtered data ----
TRAIN_OUT=$EXP_DIR/rankft_repo_heldout
echo ""
echo "=== Step 1: Train reranker (output: $TRAIN_OUT) ==="

if [ -f "$TRAIN_OUT/final/adapter_config.json" ]; then
    echo "SKIP: Training already completed ($TRAIN_OUT/final/)"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON src/train/train_rankft.py \
        --model_path $MODEL \
        --train_data $DATA_DIR/train_filtered.jsonl \
        --bm25_candidates $DATA_DIR/bm25_train_filtered.jsonl \
        --output_dir $TRAIN_OUT \
        --device cuda:0 \
        --num_epochs 2 \
        --learning_rate 5e-5 \
        --lora_rank 32 \
        --num_negatives 32 \
        --neg_bm25_ratio 0.5 \
        --neg_graph_ratio 0.25 \
        --neg_random_ratio 0.25 \
        --gradient_accumulation_steps 16 \
        --save_steps 200 \
        --seed 42

    echo "Training complete: $(date)"
fi

LORA_PATH=$TRAIN_OUT/best
if [ ! -d "$LORA_PATH" ]; then
    LORA_PATH=$TRAIN_OUT/final
fi
echo "Using LoRA: $LORA_PATH"

# ---- Step 2: Evaluate baseline on held-out test repos ----
EVAL_BASELINE=$EXP_DIR/eval_heldout_baseline
echo ""
echo "=== Step 2: Evaluate baseline on held-out repos ==="

if [ -f "$EVAL_BASELINE/summary.json" ]; then
    R1=$($PYTHON -c "import json; d=json.load(open('$EVAL_BASELINE/summary.json'))['overall']; print(f\"{d.get('recall@1', d.get('hit@1',0)):.2f}\")")
    echo "SKIP: Already done (R@1=$R1)"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA_PATH \
        --test_data $DATA_DIR/test_heldout.jsonl \
        --graph_candidates $DATA_DIR/bm25_test_heldout.jsonl \
        --output_dir $EVAL_BASELINE \
        --gpu_id 0 \
        --top_k 200 \
        --score_batch_size 2

    echo "Baseline eval complete: $(date)"
fi

# ---- Step 3: Evaluate shuffle_filenames on held-out repos ----
EVAL_SHUFFLE=$EXP_DIR/eval_heldout_shuffle_filenames
echo ""
echo "=== Step 3: Evaluate shuffle_filenames on held-out repos ==="

if [ -f "$EVAL_SHUFFLE/summary.json" ]; then
    R1=$($PYTHON -c "import json; d=json.load(open('$EVAL_SHUFFLE/summary.json'))['overall']; print(f\"{d.get('recall@1', d.get('hit@1',0)):.2f}\")")
    echo "SKIP: Already done (R@1=$R1)"
else
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON scripts/eval_rankft_4bit.py \
        --model_path $MODEL \
        --lora_path $LORA_PATH \
        --test_data $DATA_DIR/test_heldout_shuffle_filenames.jsonl \
        --graph_candidates $DATA_DIR/bm25_test_heldout_shuffle_filenames.jsonl \
        --output_dir $EVAL_SHUFFLE \
        --gpu_id 0 \
        --top_k 200 \
        --score_batch_size 2

    echo "Shuffle eval complete: $(date)"
fi

# ---- Step 4: Summary ----
echo ""
echo "=============================================="
echo "=== Results Summary ==="
echo "=============================================="

$PYTHON -c "
import json, sys

def load_summary(path):
    try:
        d = json.load(open(path))['overall']
        return {k: d.get(k, 0) for k in ['recall@1','recall@3','recall@5','recall@10','recall@20']}
    except Exception as e:
        print(f'  Could not load {path}: {e}', file=sys.stderr)
        return None

base = load_summary('$EVAL_BASELINE/summary.json')
shuf = load_summary('$EVAL_SHUFFLE/summary.json')

if base:
    print('Baseline (held-out repos, repo-heldout LoRA):')
    for k, v in base.items():
        print(f'  {k}: {v:.2f}')

if shuf:
    print('Shuffle_filenames (held-out repos, repo-heldout LoRA):')
    for k, v in shuf.items():
        print(f'  {k}: {v:.2f}')

if base and shuf:
    drop = base.get('recall@1', 0) - shuf.get('recall@1', 0)
    print(f'')
    print(f'R@1 drop from shuffle: {drop:+.2f}pp')
    if drop > 2:
        print('=> Path dependency HOLDS on unseen repos.')
    else:
        print('=> Path dependency does NOT hold on unseen repos (drop < 2pp).')
"

echo ""
echo "Done: $(date)"
