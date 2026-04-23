#!/bin/bash
# Wait for seed2 bm25 training on GPU 0 to finish, then eval.
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE

log() { echo "[$(date)] [SEED2-BM25] $1" | tee -a orchestrate.log; }

EXP="experiments/rankft_runA_bm25only_seed2"

# Wait for training to complete (final checkpoint appears)
log "Waiting for seed2 bm25 training to finish..."
while [ ! -d "${EXP}/final" ]; do sleep 120; done
log "Seed2 bm25 training done."

# Eval on merged candidates
log "Evaluating seed2 bm25 on GPU 0..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path ${EXP}/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir ${EXP}/eval_merged_rerank \
    --gpu_id 0 \
    --top_k 200 \
    --max_seq_length 512 \
    2>&1 | tee ${EXP}_eval.log
log "Seed2 bm25 eval done."

# Print result
R1=$(python3 -c "import json;d=json.load(open('${EXP}/eval_merged_rerank/summary.json'))['overall'];print(f'{d[\"recall@1\"]:.2f}')")
log "Seed2 bm25 R@1=${R1}"

# Also eval seed42 bm25 on merged candidates if not already done
SEED42="experiments/rankft_runA_bm25only"
if [ ! -f "${SEED42}/eval_merged_rerank/summary.json" ] && [ -d "${SEED42}/best" ]; then
    log "Also evaluating seed42 bm25 on merged candidates (for fair seed comparison)..."
    $PYTHON src/eval/eval_rankft.py \
        --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
        --lora_path ${SEED42}/best \
        --test_data data/grepo_text/grepo_test.jsonl \
        --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
        --output_dir ${SEED42}/eval_merged_rerank \
        --gpu_id 0 \
        --top_k 200 \
        --max_seq_length 512 \
        2>&1 | tee ${SEED42}_merged_eval.log
    R1_42=$(python3 -c "import json;d=json.load(open('${SEED42}/eval_merged_rerank/summary.json'))['overall'];print(f'{d[\"recall@1\"]:.2f}')")
    log "Seed42 bm25 (merged) R@1=${R1_42}"
fi

# Collect results
log "Collecting seed results..."
$PYTHON scripts/collect_seed_results.py
