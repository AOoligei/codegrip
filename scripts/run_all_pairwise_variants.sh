#!/bin/bash
# Orchestrate all pairwise variant experiments.
# Args: <GPU_ID>
set -e
GPU=${1:-7}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_BM25=data/rankft/swebench_bm25_final_top500.jsonl
SWE_REPOS=data/swebench_lite/repos
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_variants
mkdir -p $OUT logs

run() {
    local NAME=$1; shift
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pw_${NAME}.log | tail -3
}

# === Killer #1: SWE-bench hash_code (shuffled-path + code) ===
run swebench_hash_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant hash_code --code_lines 50

# === SWE-bench code_only (no path at all) ===
run swebench_code_only --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant code_only --code_lines 50

# === SWE-bench path_code 200 lines (extended budget) ===
run swebench_path_code_200 --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant path_code --code_lines 200

# === Code-Crucial pairwise (path) ===
CC=data/code_crucial_v2_strict_full.jsonl
GR_BM25=data/rankft/merged_bm25_exp6_candidates.jsonl
GR_REPOS=data/repos
run cc_path --test_data $CC --bm25_candidates $GR_BM25 --repo_dir $GR_REPOS --variant path

# === Code-Crucial pairwise (path_code) ===
run cc_path_code --test_data $CC --bm25_candidates $GR_BM25 --repo_dir $GR_REPOS --variant path_code --code_lines 50

# === Code-Crucial pairwise (hash_code) ===
run cc_hash_code --test_data $CC --bm25_candidates $GR_BM25 --repo_dir $GR_REPOS --variant hash_code --code_lines 50

echo "=== ALL DONE ($(date)) ==="
