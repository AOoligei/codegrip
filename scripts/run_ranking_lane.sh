#!/bin/bash
# One lane runs one (dataset, variant). $1=GPU $2=name $3..=eval args.
set -e
GPU=$1; NAME=$2; shift 2
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/ranking
mkdir -p $OUT logs
echo "=== $NAME on GPU $GPU ($(date)) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_ranking_metrics.py "$@" \
    --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/rank_${NAME}.log
