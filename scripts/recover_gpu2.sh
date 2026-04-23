#!/bin/bash
# Recovery for GPU 2: SWE-bench fair bm25 + 7B scale ablation
# The graph-hard SWE-bench fair eval is already running (will finish on its own).
# After it finishes, run bm25-only SWE-bench fair, then 7B scale.
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [GPU2-RECOVER] $1" | tee -a orchestrate.log; }

# Wait for any existing GPU 2 process to finish
log "Waiting for GPU 2 to be free..."
while nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    sleep 30
done
log "GPU 2 is free."

# Run BM25-only SWE-bench fair eval (graph-hard should be done by now)
if [ ! -f "experiments/rankft_runA_bm25only/eval_swebench_fair/summary.json" ]; then
    log "Running BM25-only SWE-bench fair eval..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path experiments/rankft_runA_bm25only/best \
        --test_data data/swebench_lite/swebench_lite_test.jsonl \
        --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
        --output_dir experiments/rankft_runA_bm25only/eval_swebench_fair \
        --gpu_id 2 \
        --top_k 50 \
        --max_seq_length 1024 \
        --score_batch_size 4 \
        2>&1 | tee experiments/rankft_runA_bm25only/eval_swebench_fair.log
    log "BM25 SWE-bench fair eval done."
else
    log "BM25 SWE-bench fair eval already done."
fi

# Compare SWE-bench results
if [ -f "experiments/rankft_runB_graph/eval_swebench_fair/summary.json" ] && \
   [ -f "experiments/rankft_runA_bm25only/eval_swebench_fair/summary.json" ]; then
    log "SWE-bench fair comparison:"
    $PYTHON -c "
import json
g = json.load(open('experiments/rankft_runB_graph/eval_swebench_fair/summary.json'))['overall']
b = json.load(open('experiments/rankft_runA_bm25only/eval_swebench_fair/summary.json'))['overall']
print(f'Graph-hard: R@1={g[\"recall@1\"]:.2f}  R@5={g[\"recall@5\"]:.2f}')
print(f'BM25-only:  R@1={b[\"recall@1\"]:.2f}  R@5={b[\"recall@5\"]:.2f}')
print(f'Delta R@1:  {g[\"recall@1\"]-b[\"recall@1\"]:+.2f}')
"
fi

# Run 7B scale ablation (clean up failed attempt first)
rm -rf experiments/scale_7B_graph 2>/dev/null
if [ ! -d "experiments/scale_7B_graph/eval_merged_rerank" ]; then
    log "Starting 7B scale ablation on GPU 2..."
    bash scripts/launch_scale_ablation.sh 7B 2 2>&1 | tee experiments/scale_7B.log
    log "7B scale ablation done."
else
    log "7B scale ablation already done."
fi

log "GPU2 recovery complete."
