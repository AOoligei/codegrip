#!/bin/bash
# Auto-launch counterfactual eval when a GPU frees up
# Checks every 2 minutes

cd /home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
CF_DATA=/data/chenlibin/grepo_agent_experiments/counterfactual/counterfactual_crossed.jsonl
CF_OUT=/data/chenlibin/grepo_agent_experiments/counterfactual/eval_path_only

# Skip if already running or done
if [ -f "$CF_OUT/summary.json" ]; then
    echo "$(date): Counterfactual eval already done"
    exit 0
fi
if pgrep -f "eval_counterfactual" > /dev/null; then
    echo "$(date): Counterfactual eval already running"
    exit 0
fi

# Find a free GPU (< 2G used)
for gpu_id in 4 6 0; do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
    if [ -n "$mem_used" ] && [ "$mem_used" -lt 2000 ]; then
        echo "$(date): Found free GPU $gpu_id (${mem_used}MiB used), launching counterfactual eval"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup $PYTHON scripts/eval_counterfactual.py \
            --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
            --lora_path experiments/rankft_runB_graph/best \
            --counterfactual_data $CF_DATA \
            --repo_dir data/repos \
            --prompt_mode path_only \
            --output_dir $CF_OUT \
            --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 4 \
            > logs/counterfactual_path_only.log 2>&1 &
        echo "$(date): Launched PID $!"
        exit 0
    fi
done
echo "$(date): No free GPU yet"
