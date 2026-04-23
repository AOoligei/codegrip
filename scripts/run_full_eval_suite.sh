#!/bin/bash
# Launch a single (lora, objective, dataset, code_mode, hash_paths) run.
# Args: GPU NAME LORA OBJECTIVE TEST BM25 REPOS [--code_mode] [--hash_paths]
set -e
GPU=$1; NAME=$2; LORA=$3; OBJ=$4; TEST=$5; BM25=$6; REPOS=$7; shift 7
EXTRA="$@"
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/ranking
mkdir -p $OUT logs
echo "=== $NAME on GPU $GPU lora=$LORA obj=$OBJ extra=$EXTRA ($(date)) ==="
ARGS="--lora_path $LORA --test_data $TEST --bm25_candidates $BM25 --repo_dir $REPOS --objective $OBJ --top_k 100 --code_lines 50 --max_len 4096 --output_dir $OUT/$NAME --gpu_id 0"
if [ "$OBJ" = "pairwise" ]; then ARGS="$ARGS --n_opponents 5"; fi
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_ranking_metrics.py $ARGS $EXTRA 2>&1 | tee logs/rank_${NAME}.log | tail -3
