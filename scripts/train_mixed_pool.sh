#!/bin/bash
# ==============================================================================
# Mixed-Pool Reranker Training Experiment
# ==============================================================================
# Addresses reviewer comment: "distribution mismatch finding should be actionable"
#
# Idea: Train reranker on negatives from BOTH BM25 and E5 dense retriever pools,
# so it generalizes across retriever distributions at test time.
#
# Negative mix (16 total per example):
#   - 25% BM25-hard   (4 negs)   -- from grepo_train_bm25_top500.jsonl
#   - 25% Dense-hard  (4 negs)   -- from grepo_train_e5large_top500.jsonl
#   - 25% Graph-hard  (4 negs)   -- co-change + import neighbors
#   - 25% Random      (4 negs)   -- uniform from repo
#
# Compare with:
#   - rankft_runA_bm25only: 50% BM25 + 0% dense + 0% graph + 50% random
#   - rankft_runB_graph:    50% BM25 + 0% dense + 25% graph + 25% random
#
# Expected outcome: mixed-pool reranker maintains R@1 on BM25+graph pool
# AND improves R@1 on hybrid+graph pool (currently drops from 27 to 20).
# ==============================================================================
set -eo pipefail

GPU_ID=${1:-2}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd /home/chenlibin/grepo_agent

MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
LORA_INIT="experiments/exp1_sft_only/stage2_sft/final"
OUTPUT_DIR="experiments/rankft_mixed_pool"

DENSE_CANDIDATES="data/rankft/grepo_train_e5large_top500.jsonl"

# ==============================================================================
# Step 0: Build E5 dense candidates for train split (if not already done)
# ==============================================================================
if [ ! -f "$DENSE_CANDIDATES" ]; then
    echo "[$(date)] Step 0: Building E5-large dense candidates for train split..."
    echo "  This takes ~20-30 min on 1 GPU."
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON \
        scripts/build_dense_train_candidates.py \
        --device cuda:0 \
        --batch_size 64 \
        --output "$DENSE_CANDIDATES" \
        2>&1 | tee "${OUTPUT_DIR}_dense_build.log"
    echo "[$(date)] Step 0 done. Dense candidates saved to $DENSE_CANDIDATES"
else
    echo "[$(date)] Step 0 skipped: $DENSE_CANDIDATES already exists."
fi

# ==============================================================================
# Step 1: Train mixed-pool reranker
# ==============================================================================
echo ""
echo "[$(date)] Step 1: Training mixed-pool reranker..."
echo "  GPU: $GPU_ID"
echo "  Neg mix: BM25=0.25, Dense=0.25, Graph=0.25, Random=0.25"
echo "  Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 \
$PYTHON src/train/train_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_INIT" \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dense_candidates "$DENSE_CANDIDATES" \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --num_negatives 16 \
    --neg_bm25_ratio 0.25 \
    --neg_dense_ratio 0.25 \
    --neg_graph_ratio 0.25 \
    --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --max_seq_length 512 \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[$(date)] Step 1 done. Training complete."

# ==============================================================================
# Step 2: Evaluate on BM25+graph pool (standard setting)
# ==============================================================================
BEST_DIR="$OUTPUT_DIR/best"
if [ ! -d "$BEST_DIR" ]; then
    BEST_DIR="$OUTPUT_DIR/final"
fi

echo ""
echo "[$(date)] Step 2: Evaluating on BM25+graph pool..."
EVAL_BM25G="$OUTPUT_DIR/eval_bm25_graph"
mkdir -p "$EVAL_BM25G"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_both_edge_types_candidates.jsonl \
    --output_dir "$EVAL_BM25G" \
    --gpu_id 0 --top_k 200 --max_seq_length 512 \
    2>&1 | tee "$EVAL_BM25G/eval.log"

R1_BG=$(python3 -c "import json; print(f'{json.load(open(\"$EVAL_BM25G/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Mixed-pool on BM25+graph: R@1=$R1_BG"

# ==============================================================================
# Step 3: Evaluate on hybrid+graph pool (the distribution-mismatch setting)
# ==============================================================================
echo ""
echo "[$(date)] Step 3: Evaluating on hybrid+graph pool..."
EVAL_HYB="$OUTPUT_DIR/eval_hybrid_graph"
mkdir -p "$EVAL_HYB"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_hybrid_matched_graph_candidates.jsonl \
    --output_dir "$EVAL_HYB" \
    --gpu_id 0 --top_k 200 --max_seq_length 512 \
    2>&1 | tee "$EVAL_HYB/eval.log"

R1_HG=$(python3 -c "import json; print(f'{json.load(open(\"$EVAL_HYB/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Mixed-pool on hybrid+graph: R@1=$R1_HG"

# ==============================================================================
# Step 4: Evaluate on BM25-only pool (for completeness)
# ==============================================================================
echo ""
echo "[$(date)] Step 4: Evaluating on BM25-only pool..."
EVAL_BM25="$OUTPUT_DIR/eval_bm25pool"
mkdir -p "$EVAL_BM25"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 $PYTHON src/eval/eval_rankft.py \
    --model_path "$MODEL_PATH" \
    --lora_path "$BEST_DIR" \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
    --output_dir "$EVAL_BM25" \
    --gpu_id 0 --top_k 200 --max_seq_length 512 \
    2>&1 | tee "$EVAL_BM25/eval.log"

R1_B=$(python3 -c "import json; print(f'{json.load(open(\"$EVAL_BM25/summary.json\"))[\"overall\"][\"recall@1\"]:.2f}')")
echo "[$(date)] Mixed-pool on BM25-only: R@1=$R1_B"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "================================================================"
echo "  MIXED-POOL RERANKER RESULTS"
echo "================================================================"
echo ""
echo "  Mixed-pool reranker (this experiment):"
echo "    BM25+graph pool:     R@1 = $R1_BG"
echo "    Hybrid+graph pool:   R@1 = $R1_HG"
echo "    BM25-only pool:      R@1 = $R1_B"
echo ""
echo "  Baselines (from existing runs):"
echo "    runB_graph on BM25+graph:    R@1 = 27.01"
echo "    runB_graph on hybrid+graph:  R@1 = 20.00  <-- distribution mismatch!"
echo "    runA_bm25only on BM25-only:  R@1 = 19.00"
echo ""
echo "  Goal: mixed-pool should close the gap on hybrid+graph"
echo "  (ideally R@1 > 24 on hybrid) while maintaining on BM25+graph."
echo "================================================================"
