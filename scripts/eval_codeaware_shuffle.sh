#!/bin/bash
# Evaluate rankft_codeaware_swetrain under path hashing vs normal, + baseline comparison
GPU=${1:-0}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent

NEW_LORA=experiments/rankft_codeaware_swetrain/best
OLD_LORA=experiments/rankft_runB_graph/best
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_BM25=data/rankft/swebench_bm25_final_top500.jsonl
SWEV_TEST=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_prepared.jsonl
SWEV_BM25=/data/chenlibin/grepo_agent_experiments/swebench_verified/swebench_verified_bm25_top500.jsonl
REPOS=data/swebench_lite/repos
OUT=/data/chenlibin/grepo_agent_experiments/ranking

run() {
    local NAME=$1; shift
    local OD=$OUT/$NAME
    if [ -f "$OD/summary.json" ]; then echo "SKIP $NAME (exists)"; return; fi
    echo "=== $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_ranking_metrics.py \
        "$@" --output_dir $OD --gpu_id 0 2>&1 | tee logs/ca_${NAME}.log | tail -5
}

# codeaware on SWE-bench Lite hash
run newlora_swe_point_code_hash \
    --lora_path $NEW_LORA --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $REPOS \
    --objective pointwise --code_mode --hash_paths --top_k 100 --code_lines 50

# codeaware on SWE-bench Verified hash
run newlora_swev_point_code_hash \
    --lora_path $NEW_LORA --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $REPOS \
    --objective pointwise --code_mode --hash_paths --top_k 100 --code_lines 50

# baseline on SWE-bench Lite hash (for comparison)
run oldlora_swe_point_code_hash \
    --lora_path $OLD_LORA --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $REPOS \
    --objective pointwise --code_mode --hash_paths --top_k 100 --code_lines 50

# baseline on SWE-bench Verified hash
run oldlora_swev_point_code_hash \
    --lora_path $OLD_LORA --test_data $SWEV_TEST --bm25_candidates $SWEV_BM25 --repo_dir $REPOS \
    --objective pointwise --code_mode --hash_paths --top_k 100 --code_lines 50

echo "=== DONE ($(date)) ==="
