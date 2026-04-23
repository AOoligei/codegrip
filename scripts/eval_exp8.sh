#!/bin/bash
# Evaluate Exp 8 (Graph-conditioned SFT) with graph prompt mode + expansion + reranking
# Usage: bash scripts/eval_exp8.sh [gpu_id]

PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"
BASE_DIR="/home/chenlibin/grepo_agent"
cd "$BASE_DIR"

GPU="${1:-4}"
MODEL_PATH="/data/shuyang/models/Qwen2.5-7B-Instruct"
EXP_NAME="exp8_graph_sft"
ADAPTER_PATH="experiments/$EXP_NAME/stage2_sft/final"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    exit 1
fi

# Step 1: Base evaluation with GRAPH prompt mode (matches training format)
if [ ! -f "experiments/${EXP_NAME}/eval_graph/predictions.jsonl" ]; then
    echo "=== Evaluating $EXP_NAME (graph mode) on GPU $GPU ==="
    mkdir -p "experiments/${EXP_NAME}/eval_graph"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$ADAPTER_PATH" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir "experiments/${EXP_NAME}/eval_graph" \
        --prompt_mode graph \
        --file_tree_dir data/file_trees \
        --dep_graph_dir data/dep_graphs \
        --train_data data/grepo_text/grepo_train.jsonl \
        2>&1 | tee "experiments/${EXP_NAME}/eval_graph.log"
fi

# Also eval with filetree mode for comparison (does graph training help even without graph prompt?)
if [ ! -f "experiments/${EXP_NAME}/eval_filetree/predictions.jsonl" ]; then
    echo "=== Evaluating $EXP_NAME (filetree mode, for comparison) on GPU $GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$MODEL_PATH" \
        --lora_path "$ADAPTER_PATH" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output_dir "experiments/${EXP_NAME}/eval_filetree" \
        --prompt_mode filetree \
        --file_tree_dir data/file_trees \
        2>&1 | tee "experiments/${EXP_NAME}/eval_filetree.log"
fi

# Step 2: Expansion on the graph-mode predictions
PRED_DIR="eval_graph"
if [ -f "experiments/${EXP_NAME}/${PRED_DIR}/predictions.jsonl" ] && \
   [ ! -f "experiments/${EXP_NAME}/eval_graph_expansion/predictions.jsonl" ]; then
    echo "=== Running expansion for $EXP_NAME ==="
    mkdir -p "experiments/${EXP_NAME}/eval_graph_expansion"
    $PYTHON src/eval/multi_signal_expansion.py \
        --predictions "experiments/${EXP_NAME}/${PRED_DIR}/predictions.jsonl" \
        --train_data data/grepo_text/grepo_train.jsonl \
        --dep_graph_dir data/dep_graphs \
        --file_tree_dir data/file_trees \
        --output "experiments/${EXP_NAME}/eval_graph_expansion/predictions.jsonl" \
        --max_expand 35 --min_cochange_score 0.02 --max_dir_size 35
fi

# Step 3: Reranking
if [ -f "experiments/${EXP_NAME}/eval_graph_expansion/predictions.jsonl" ] && \
   [ ! -f "experiments/${EXP_NAME}/eval_graph_reranked/predictions.jsonl" ]; then
    echo "=== Running reranking for $EXP_NAME ==="
    mkdir -p "experiments/${EXP_NAME}/eval_graph_reranked"
    $PYTHON src/eval/rerank_predictions.py \
        --predictions "experiments/${EXP_NAME}/eval_graph_expansion/predictions.jsonl" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --output "experiments/${EXP_NAME}/eval_graph_reranked/predictions.jsonl" \
        --promote_top5 1 --promote_top10 3 --threshold 0.2
fi

echo
echo "=== Results Summary ==="
for stage in eval_graph eval_filetree eval_graph_expansion eval_graph_reranked; do
    f="experiments/${EXP_NAME}/${stage}/summary.json"
    if [ -f "$f" ]; then
        echo -n "$(printf '%-40s' "${EXP_NAME} (${stage})")"
        $PYTHON -c "
import json
with open('$f') as f:
    o = json.load(f)['overall']
print(f\"H@1={o.get('hit@1',0):6.2f}  H@5={o.get('hit@5',0):6.2f}  H@10={o.get('hit@10',0):6.2f}  H@20={o.get('hit@20',0):6.2f}\")
"
    fi
done
echo "GAT baseline:                            H@1= 14.80  H@5= 31.51  H@10= 37.40  H@20= 41.25"
echo "Exp1 best (SFT+expand+rerank):           H@1= 18.73  H@5= 33.24  H@10= 38.57  H@20= 42.79"
