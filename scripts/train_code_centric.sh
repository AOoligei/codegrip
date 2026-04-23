#!/bin/bash
# Train code-centric reranker: real paths + code content
# Addresses reviewer criticism: "you haven't tested a truly strong code-reading baseline"
#
# Key differences from path-only (train_delex_reranker.sh):
#   - Prompt includes code content (first 50 lines + function signatures)
#   - num_negatives=4 (not 16) to fit code in 1024-token window
#   - max_seq_length=1024 (not 512) to accommodate code
#   - No anonymization — real paths + real code
#
# Key differences from code-residual:
#   - Real paths (not anonymized) — tests if code ADDS signal on top of paths
#   - Better code extraction (AST signatures, not just first N lines)
#   - Same LoRA init, same data, same compute as path-only baseline
#
# GPU memory estimate (1024 seq_len, batch=1, grad_accum=16, 4 negs):
#   Model: ~14GB (7B bf16), LoRA grads: ~1GB, KV cache: ~1GB
#   Forward: 5 prompts * 1024 tokens * ~2MB ≈ 10GB peak
#   Total: ~26GB — fits on RTX 4090 (24GB) with gradient checkpointing
#   If OOM: reduce code_max_chars to 1000 or num_negatives to 3

GPU_ID=${1:-3}
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python3
OUTPUT_DIR=/data/chenlibin/grepo_agent_experiments/code_centric_scorer

echo "=== Code-Centric Reranker Training (GPU $GPU_ID) ==="
echo "Start: $(date)"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u src/train/train_rankft_code_centric.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --repo_dir data/repos \
    --output_dir $OUTPUT_DIR \
    --device cuda:0 \
    --num_negatives 4 \
    --neg_bm25_ratio 0.5 \
    --neg_graph_ratio 0.25 \
    --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --logging_steps 10 \
    --max_seq_length 1024 \
    --lora_rank 32 \
    --seed 42 \
    --code_head_lines 50 \
    --code_max_chars 1500

echo "End: $(date)"
