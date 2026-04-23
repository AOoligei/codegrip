#!/bin/bash
# Full ranking sweep: pointwise +/-code, pairwise +/-code on 4 datasets.
# Uses Codex-approved scripts/eval_ranking_metrics.py
set -e
GPU=${1:-0}
TAG=${2:-v3}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/ranking
mkdir -p $OUT logs

run () {
    local NAME=$1; shift
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_ranking_metrics.py "$@" \
        --output_dir $OUT/${TAG}_${NAME} --gpu_id 0 2>&1 | tee logs/rank_${TAG}_${NAME}.log | tail -3
}

# ===== Pointwise =====
# SWE-Verified
SWEV_TEST=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_prepared.jsonl
SWEV_BM25=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_bm25_top500.jsonl
SWEV_REPOS=data/swebench_lite/repos
run swev_point      --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --objective pointwise --top_k 100
run swev_point_code --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --objective pointwise --code_mode --top_k 100

# SWE-Lite
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_BM25=data/rankft/swebench_bm25_final_top500.jsonl
SWE_REPOS=data/swebench_lite/repos
run swe_point      --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --objective pointwise --top_k 100
run swe_point_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --objective pointwise --code_mode --top_k 100

# GREPO codeavail
G_TEST=data/grepo_text/grepo_test_codeavail.jsonl
G_BM25=data/rankft/merged_bm25_exp6_candidates.jsonl
G_REPOS=data/repos
run grepo_point      --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --objective pointwise --top_k 100
run grepo_point_code --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --objective pointwise --code_mode --top_k 100

# Code-Crucial
CC=data/code_crucial_v2_strict_full.jsonl
run cc_point      --test_data $CC --bm25_candidates $G_BM25 --repo_dir $G_REPOS --objective pointwise --top_k 100
run cc_point_code --test_data $CC --bm25_candidates $G_BM25 --repo_dir $G_REPOS --objective pointwise --code_mode --top_k 100

echo "=== ALL DONE ($(date)) ==="
