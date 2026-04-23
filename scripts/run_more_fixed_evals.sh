#!/bin/bash
set -e
GPU=${1:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_fixed
mkdir -p $OUT logs

run () {
    local NAME=$1; shift
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pwfix_${NAME}.log | tail -3
}

# === GREPO code-available subset ===
G_TEST=data/grepo_text/grepo_test_codeavail.jsonl
G_BM25=data/rankft/merged_bm25_exp6_candidates.jsonl
G_REPOS=data/repos
run grepo_avail_path      --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path --miss_abort_frac 0.30
run grepo_avail_path_code --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path_code --code_lines 50 --miss_abort_frac 0.30
run grepo_avail_hash_code --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant hash_code --code_lines 50 --miss_abort_frac 0.30

# === SWE-bench Verified ===
SWEV_TEST=data/swebench_verified/swebench_verified_test.jsonl
SWEV_BM25=data/swebench_verified/swebench_verified_bm25_candidates.jsonl
SWEV_REPOS=data/swebench_verified/repos
if [ -f "$SWEV_TEST" ]; then
    run swev_path      --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --variant path
    run swev_path_code --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $SWEV_REPOS --variant path_code --code_lines 50
else
    echo "(SWE-Verified data not found, skipping)"
fi

echo "=== ALL DONE ($(date)) ==="
