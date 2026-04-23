#!/bin/bash
# Re-run all pairwise variants with the SHUFFLE-file candidate
# (same as old script / paper baseline) for apples-to-apples comparison.
set -e
GPU=${1:-4}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_CAND=data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl
SWE_REPOS=data/swebench_lite/repos
REAL_BM25=data/rankft/swebench_bm25_final_top500.jsonl
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_variants_v2
mkdir -p $OUT logs

run() {
    local NAME=$1; shift
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pwv2_${NAME}.log | tail -3
}

# === Apples-to-apples vs paper's 86% (shuffle-file candidates) ===
run swe_path        --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant path
run swe_path_code   --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant path_code --code_lines 50
run swe_hash_code   --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant hash_code --code_lines 50
run swe_code_only   --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant code_only --code_lines 50
run swe_path_code_200 --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant path_code --code_lines 200

# === Robustness: real BM25 (harder hard-neg) ===
run robust_real_bm25_path      --test_data $SWE_TEST --bm25_candidates $REAL_BM25 --repo_dir $SWE_REPOS --variant path
run robust_real_bm25_path_code --test_data $SWE_TEST --bm25_candidates $REAL_BM25 --repo_dir $SWE_REPOS --variant path_code --code_lines 50

echo "=== ALL DONE ($(date)) ==="
