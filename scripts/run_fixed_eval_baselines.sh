#!/bin/bash
# Re-run baselines with FIXED eval script (correct " A"/" B" tokens, deterministic GT, etc).
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

# === SWE-bench Lite ===
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_BM25=data/rankft/swebench_bm25_final_top500.jsonl
SWE_REPOS=data/swebench_lite/repos
run swe_path      --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant path
run swe_path_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant path_code --code_lines 50
run swe_hash_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant hash_code --code_lines 50
run swe_code_only --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant code_only --code_lines 50

# === GREPO ===
G_TEST=data/grepo_text/grepo_test.jsonl
G_BM25=data/rankft/merged_bm25_exp6_candidates.jsonl
G_REPOS=data/repos
run grepo_path      --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path
run grepo_path_code --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path_code --code_lines 50
run grepo_hash_code --test_data $G_TEST --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant hash_code --code_lines 50

# === Code-Crucial (GREPO subset, 116 ex) ===
CC=data/code_crucial_v2_strict_full.jsonl
run cc_path      --test_data $CC --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path
run cc_path_code --test_data $CC --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant path_code --code_lines 50
run cc_hash_code --test_data $CC --bm25_candidates $G_BM25 --repo_dir $G_REPOS --variant hash_code --code_lines 50

echo "=== ALL DONE ($(date)) ==="
