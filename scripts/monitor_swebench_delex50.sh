#!/bin/bash
set -euo pipefail
# Monitor swebench_adapted_delex50 training, then auto-eval on completion
GPU_ID=${1:-3}
TRAIN_DIR="experiments/swebench_adapted_delex50"
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL="/data/shuyang/models/Qwen2.5-7B-Instruct"
cd /home/chenlibin/grepo_agent

echo "[$(date)] Monitoring ${TRAIN_DIR} training..."

while [ ! -f "${TRAIN_DIR}/final/adapter_config.json" ] && [ ! -f "${TRAIN_DIR}/final/adapter_model.safetensors" ]; do
    sleep 120
    if [ -f "${TRAIN_DIR}/training_diagnostics.jsonl" ]; then
        LAST=$(tail -1 "${TRAIN_DIR}/training_diagnostics.jsonl" 2>/dev/null)
        STATUS=$(echo "$LAST" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"step {d.get(\"step\",\"?\")}, loss {d.get(\"loss\",\"?\")}")' 2>/dev/null || echo "step ?, loss ?")
        echo "[$(date)] ${STATUS}"
    fi
done

echo "[$(date)] Training complete! Launching eval on GPU ${GPU_ID}..."

LORA_DIR=$(${PYTHON} scripts/experiment_automation.py resolve-adapter --exp-dir "${TRAIN_DIR}" 2>/dev/null || true)
if [ -z "${LORA_DIR}" ]; then
    echo "[$(date)] Could not resolve adapter dir for ${TRAIN_DIR}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} src/eval/eval_rankft.py \
    --model_path ${MODEL} \
    --lora_path ${LORA_DIR} \
    --test_data data/swebench_lite/swebench_lite_test.jsonl \
    --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
    --output_dir ${TRAIN_DIR}/eval_swebench_best \
    --quantization 4bit-nf4 \
    --top_k 50 \
    --max_seq_length 1024

echo "[$(date)] Eval complete!"
cat ${TRAIN_DIR}/eval_swebench_best/summary.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
o = d['overall']
print(f\"  H@1:  {o['hit@1']:.2f}%\")
print(f\"  H@3:  {o['hit@3']:.2f}%\")
print(f\"  H@5:  {o['hit@5']:.2f}%\")
print(f\"  H@10: {o['hit@10']:.2f}%\")
" || echo "  Summary unavailable"
