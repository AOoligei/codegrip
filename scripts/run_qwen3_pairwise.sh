#!/bin/bash
set -e
GPU=${1:-1}
PY=/home/chenlibin/miniconda3/envs/tgn/bin/python3
cd /home/chenlibin/grepo_agent
SWE_TEST=data/swebench_lite/swebench_lite_test.jsonl
SWE_BM25=data/rankft/swebench_bm25_final_top500.jsonl
SWE_REPOS=data/swebench_lite/repos
OUT=/data/chenlibin/grepo_agent_experiments/pairwise_qwen3
mkdir -p $OUT logs

run () {
    local NAME=$1; shift
    echo "=== qwen3 $NAME ($(date)) ==="
    CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/eval_pairwise_variants.py "$@" \
        --model_path /data/hzy/models/Qwen3-8B \
        --lora_path experiments/cross_llm_qwen3_8b/best \
        --output_dir $OUT/$NAME --gpu_id 0 2>&1 | tee logs/pw_qwen3_${NAME}.log | tail -3
}

run swe_path      --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant path
run swe_path_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant path_code --code_lines 50
run swe_hash_code --test_data $SWE_TEST --bm25_candidates $SWE_BM25 --repo_dir $SWE_REPOS --variant hash_code --code_lines 50

echo "=== qwen3 DONE ($(date)) ==="
