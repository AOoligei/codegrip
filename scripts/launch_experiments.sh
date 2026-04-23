#!/bin/bash
# Launch CodeGRIP training experiments on specified GPUs
# Usage: bash scripts/launch_experiments.sh <exp_name> <gpu_id>

set -e

EXP_NAME=$1
GPU_ID=$2

if [ -z "$EXP_NAME" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <exp_name> <gpu_id>"
    echo ""
    echo "Available experiments:"
    for d in experiments/exp*; do
        if [ -f "$d/config.json" ]; then
            name=$(basename $d)
            has_final="no"
            [ -d "$d/stage2_sft/final" ] && has_final="yes"
            echo "  $name (done=$has_final)"
        fi
    done
    exit 1
fi

EXP_DIR="experiments/$EXP_NAME"
CONFIG="$EXP_DIR/config.json"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found"
    exit 1
fi

# Check if already done
if [ -d "$EXP_DIR/stage2_sft/final" ]; then
    echo "Warning: $EXP_NAME already has a final checkpoint"
    read -p "Re-run? (y/n) " confirm
    [ "$confirm" != "y" ] && exit 0
fi

# Extract config values
MODEL_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['model_path'])")
SFT_DATA=$(python3 -c "import json; print(json.load(open('$CONFIG'))['sft_data'])")
GSP_DATA=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(c.get('gsp_data') or '')")
SKIP_GSP=$(python3 -c "import json; print('--skip_gsp' if json.load(open('$CONFIG')).get('skip_gsp', True) else '')")
SKIP_SFT=$(python3 -c "import json; print('--skip_sft' if json.load(open('$CONFIG')).get('skip_sft', False) else '')")

echo "=== Launching $EXP_NAME on GPU $GPU_ID ==="
echo "  Model: $MODEL_PATH"
echo "  SFT data: $SFT_DATA"
echo "  GSP data: $GSP_DATA"

CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python src/train/train_codegrip.py \
    --model_path $MODEL_PATH \
    --sft_data $SFT_DATA \
    --output_dir $EXP_DIR \
    --device cuda:0 \
    $SKIP_GSP $SKIP_SFT"

if [ -n "$GSP_DATA" ]; then
    CMD="$CMD --gsp_data $GSP_DATA"
fi

echo "  Command: $CMD"
echo ""

# Launch with nohup
nohup bash -c "$CMD" > "$EXP_DIR/train.log" 2>&1 &
PID=$!
echo "  Launched PID: $PID"
echo "  Log: $EXP_DIR/train.log"
echo "$PID" > "$EXP_DIR/train.pid"
