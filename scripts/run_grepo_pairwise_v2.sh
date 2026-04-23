#!/bin/bash
# Correct GREPO pairwise on real BM25 + real code
set -e
GPU=${1:-5}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
TEST=data/grepo_text/grepo_test.jsonl
BM25=data/rankft/merged_bm25_exp6_candidates.jsonl
REPOS=data/repos
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_grepo_v2
mkdir -p $OUT logs

run () {
    local NAME=$1; shift
    echo "=== grepo $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --test_data $TEST --bm25_candidates $BM25 --repo_dir $REPOS \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pw_grepo_${NAME}.log | tail -3
}

run path       --variant path
run path_code  --variant path_code --code_lines 50
run hash_code  --variant hash_code --code_lines 50
run code_only  --variant code_only --code_lines 50

echo "=== DONE ($(date)) ==="
