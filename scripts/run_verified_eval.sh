#!/bin/bash
set -e
GPU=${1:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_fixed
SWEV_TEST=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_prepared.jsonl
SWEV_BM25=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_bm25_top500.jsonl
SWEV_REPOS=data/swebench_lite/repos

run () {
    local NAME=$1; shift
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pwfix_${NAME}.log | tail -3
}

run swev_path      --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --variant path
run swev_path_code --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --variant path_code --code_lines 50
run swev_hash_code --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --variant hash_code --code_lines 50

echo "=== ALL DONE ($(date)) ==="
