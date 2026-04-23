#!/bin/bash
# Wait for 7B bm25 training on GPU 6 to finish, then eval
set -eo pipefail
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent
TRAIN_PID=2173701

echo "[$(date)] Waiting for 7B bm25 training (PID $TRAIN_PID) to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; done
echo "[$(date)] 7B bm25 training done."

if [ ! -d "experiments/scale_7B_bm25only/final" ]; then
    echo "[$(date)] ERROR: no final checkpoint found"
    exit 1
fi

echo "[$(date)] Starting 7B bm25 eval on GPU 6..."
CUDA_VISIBLE_DEVICES=6 $PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/scale_7B_bm25only/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_7B_bm25only/eval_merged_rerank \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee experiments/scale_7B_bm25only_eval.log

echo "[$(date)] 7B bm25 eval done."
python3 -c "
import json
d = json.load(open('experiments/scale_7B_bm25only/eval_merged_rerank/summary.json'))['overall']
print(f'7B BM25-only R@1={d[\"recall@1\"]:.2f}')
"
