#!/bin/bash
# Evaluate all completed experiments that don't have eval results yet
# Usage: bash scripts/eval_all_pending.sh <gpu_id>
set -e
cd /home/chenlibin/grepo_agent

GPU_ID=${1:-0}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python

echo "=== Evaluating all pending experiments on GPU $GPU_ID ==="

# List of experiments to evaluate
for exp_dir in experiments/exp*/; do
    exp_name=$(basename $exp_dir)
    final_dir="$exp_dir/stage2_sft/final"
    eval_dir="$exp_dir/eval_filetree"
    
    # Skip if no final checkpoint
    [ ! -d "$final_dir" ] && continue
    
    # Skip if already evaluated
    [ -f "$eval_dir/summary.json" ] && continue
    
    # Get model path from config
    config="$exp_dir/config.json"
    [ ! -f "$config" ] && continue
    model_path=$(python3 -c "import json; print(json.load(open('$config'))['model_path'])")
    
    echo ""
    echo "=== Evaluating $exp_name ==="
    echo "  Model: $model_path"
    echo "  LoRA: $final_dir"
    
    mkdir -p "$eval_dir"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON src/eval/eval_grepo_file_level.py \
        --model_path "$model_path" \
        --lora_path "$final_dir" \
        --test_data data/grepo_text/grepo_test.jsonl \
        --file_tree_dir data/file_trees \
        --output_dir "$eval_dir" \
        --prompt_mode filetree 2>&1 | tee "$eval_dir/eval.log"
    
    echo "  Done: $exp_name"
    
    # Also run SWE-bench eval
    swebench_dir="$exp_dir/eval_swebench_lite"
    if [ ! -f "$swebench_dir/summary.json" ]; then
        echo "  Running SWE-bench Lite eval for $exp_name..."
        mkdir -p "$swebench_dir"
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON src/eval/eval_grepo_file_level.py \
            --model_path "$model_path" \
            --lora_path "$final_dir" \
            --test_data data/swebench_lite/swebench_lite_test.jsonl \
            --file_tree_dir data/swebench_lite/file_trees \
            --output_dir "$swebench_dir" \
            --prompt_mode filetree 2>&1 | tee "$swebench_dir/eval.log"
        echo "  Done SWE-bench: $exp_name"
    fi
done

echo ""
echo "=== All evaluations complete ==="

# Print summary table
echo ""
echo "=== Results Summary ==="
printf "%-35s %8s %8s %8s %8s\n" "Experiment" "Hit@1" "Hit@5" "Hit@10" "Hit@20"
echo "------------------------------------------------------------------------"
for exp_dir in experiments/exp*/eval_filetree/; do
    if [ -f "$exp_dir/summary.json" ]; then
        exp_name=$(basename $(dirname $exp_dir))
        python3 -c "
import json
with open('${exp_dir}summary.json') as f:
    s = json.load(f)
o = s.get('overall', s.get('metrics', {}))
print(f'$exp_name'.ljust(35), end='')
for k in ['hit@1','hit@3','hit@5','hit@10','hit@20']:
    v = o.get(k, 0)
    print(f'{v:8.2f}', end='')
print()
"
    fi
done
