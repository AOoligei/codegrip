#!/bin/bash
# Comprehensive RankFT evaluation across all runs and benchmarks
# Usage: bash scripts/eval_all_rankft.sh <GPU_ID>
GPU_ID=${1:-5}
cd /home/chenlibin/grepo_agent

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"

eval_run() {
    local run_name=$1
    local lora_path=$2
    local label=$3
    local eval_script=${4:-"src/eval/eval_rankft.py"}
    local grepo_extra_args=$5
    local swebench_extra_args=$6

    if [ ! -d "$lora_path" ]; then
        echo "  [skip] $run_name: $lora_path not found"
        return
    fi

    # GREPO test (K=200)
    if [ ! -f "experiments/${label}_grepo_k200/summary.json" ]; then
        echo "=== $run_name on GREPO (K=200) ==="
        CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 \
        $PYTHON $eval_script \
            --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
            --lora_path "$lora_path" \
            --test_data data/grepo_text/grepo_test.jsonl \
            --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
            --output_dir "experiments/${label}_grepo_k200" \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16 \
            $grepo_extra_args
    fi

    # SWE-bench (K=200)
    if [ ! -f "experiments/${label}_swebench_k200/summary.json" ]; then
        echo "=== $run_name on SWE-bench (K=200) ==="
        CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 \
        $PYTHON $eval_script \
            --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
            --lora_path "$lora_path" \
            --test_data data/swebench_lite/swebench_lite_test.jsonl \
            --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
            --output_dir "experiments/${label}_swebench_k200" \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16 \
            $swebench_extra_args
    fi
}

# Path-only runs (use eval_rankft.py)
eval_run "Run A (BM25-hard)" "experiments/rankft_runA_bm25only/final" "rankft_runA"
eval_run "Run A (BM25-hard, best)" "experiments/rankft_runA_bm25only/best" "rankft_runA_best"
eval_run "Run B (Graph neg)" "experiments/rankft_runB_graph/final" "rankft_runB"
eval_run "Run B (Graph neg, best)" "experiments/rankft_runB_graph/best" "rankft_runB_best"
eval_run "Run C (Random neg)" "experiments/rankft_runC_random/final" "rankft_runC"

# Content-aware runs (use eval_rankft_content.py with proper summary files)
eval_run "Run D (Content)" "experiments/rankft_runD_content/final" "rankft_runD" \
    "src/eval/eval_rankft_content.py" \
    "--file_summaries data/file_summaries_aligned.json" \
    "--file_summaries data/swebench_file_summaries/file_summaries_all.json"

eval_run "Run D (Content, best)" "experiments/rankft_runD_content/best" "rankft_runD_best" \
    "src/eval/eval_rankft_content.py" \
    "--file_summaries data/file_summaries_aligned.json" \
    "--file_summaries data/swebench_file_summaries/file_summaries_all.json"

eval_run "Run E (Content fresh)" "experiments/rankft_runE_content_fresh/final" "rankft_runE" \
    "src/eval/eval_rankft_content.py" \
    "--file_summaries data/file_summaries_aligned.json" \
    "--file_summaries data/swebench_file_summaries/file_summaries_all.json"

echo ""
echo "=== Summary ==="
for d in experiments/rankft_run*_grepo_k200 experiments/rankft_run*_swebench_k200; do
    if [ -f "$d/summary.json" ]; then
        echo "--- $(basename $d) ---"
        $PYTHON -c "
import json
s = json.load(open('$d/summary.json'))
o = s['overall']
print(f\"  Hit@1: {o.get('hit@1',0):.2f}%  Hit@5: {o.get('hit@5',0):.2f}%  Hit@10: {o.get('hit@10',0):.2f}%\")
print(f\"  Acc@1: {o.get('acc@1',0):.2f}%  Cond.Acc@1: {o.get('cond_acc@1|gt_in_candidates', s.get('cond_acc1',0)):.2f}%\")
"
    fi
done
