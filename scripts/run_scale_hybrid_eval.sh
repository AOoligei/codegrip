#!/bin/bash
# Eval scale models (0.5B, 1.5B, 3B) on hybrid pool to test oracle fallacy across scales
# Run sequentially on one GPU to avoid OOM

GPU_ID=${1:-0}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
TEST_DATA=data/grepo_text/grepo_test.jsonl
HYBRID_CANDIDATES=data/rankft/merged_hybrid_e5large_graph_candidates.jsonl

echo "=== Scale Oracle Fallacy Eval (GPU $GPU_ID) ==="
echo "Start: $(date)"

# 0.5B on hybrid pool
echo ""
echo ">>> [0.5B hybrid] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
  --model_path /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct \
  --lora_path experiments/scale_0.5B_graph/final \
  --test_data $TEST_DATA \
  --bm25_candidates $HYBRID_CANDIDATES \
  --output_dir experiments/scale_0.5B_graph/eval_hybrid \
  --gpu_id 0 \
  --top_k 200 \
  --max_seq_length 512 \
  --score_batch_size 32
echo "  0.5B hybrid done: $(date)"

# 1.5B on hybrid pool
echo ""
echo ">>> [1.5B hybrid] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
  --model_path /data/chenlibin/models/Qwen2.5-1.5B-Instruct \
  --lora_path experiments/scale_1.5B_graph/final \
  --test_data $TEST_DATA \
  --bm25_candidates $HYBRID_CANDIDATES \
  --output_dir experiments/scale_1.5B_graph/eval_hybrid \
  --gpu_id 0 \
  --top_k 200 \
  --max_seq_length 512 \
  --score_batch_size 32
echo "  1.5B hybrid done: $(date)"

# 3B on hybrid pool
echo ""
echo ">>> [3B hybrid] $(date)"
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u scripts/eval_rankft_4bit.py \
  --model_path /data/chenlibin/models/Qwen2.5-3B-Instruct \
  --lora_path experiments/scale_3B_graph/final \
  --test_data $TEST_DATA \
  --bm25_candidates $HYBRID_CANDIDATES \
  --output_dir experiments/scale_3B_graph/eval_hybrid \
  --gpu_id 0 \
  --top_k 200 \
  --max_seq_length 512 \
  --score_batch_size 16
echo "  3B hybrid done: $(date)"

echo ""
echo "=== All Done ==="
echo "End: $(date)"

# Print summary
echo ""
echo "=== Summary ==="
for size in 0.5B 1.5B 3B; do
  graph_f="experiments/scale_${size}_graph/eval_merged_rerank/summary.json"
  hybrid_f="experiments/scale_${size}_graph/eval_hybrid/summary.json"
  if [ -f "$graph_f" ] && [ -f "$hybrid_f" ]; then
    $PYTHON -c "
import json
g = json.load(open('$graph_f'))['overall']
h = json.load(open('$hybrid_f'))['overall']
gr1 = g.get('recall@1', g.get('hit@1', 0))
hr1 = h.get('recall@1', h.get('hit@1', 0))
print(f'${size}: graph R@1={gr1:.2f}, hybrid R@1={hr1:.2f}, delta={hr1-gr1:.2f}')
"
  fi
done
