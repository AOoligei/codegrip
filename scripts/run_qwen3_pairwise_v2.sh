#!/bin/bash
set -e
GPU=${1:-5}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_CAND=data/swebench_lite/swebench_perturb_shuffle_filenames_candidates.jsonl
SWE_REPOS=data/swebench_lite/repos
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_qwen3_v2
mkdir -p $OUT logs
run () {
    local NAME=$1; shift
    echo "=== qwen3 v2 $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --model_path /data/hzy/models/Qwen3-8B \
        --lora_path experiments/cross_llm_qwen3_8b/best \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pwv2_qwen3_${NAME}.log | tail -3
}
run swe_path      --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant path
run swe_path_code --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant path_code --code_lines 50
run swe_hash_code --test_data $SWE_TEST --bm25_candidates $SWE_CAND --repo_dir $SWE_REPOS --variant hash_code --code_lines 50
echo "=== qwen3 v2 DONE ($(date)) ==="
