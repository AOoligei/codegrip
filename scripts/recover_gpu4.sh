#!/bin/bash
set -e
BASE=/home/chenlibin/grepo_agent
PYTHON=/home/chenlibin/miniconda3/envs/tgn/bin/python
cd $BASE
log() { echo "[$(date)] [GPU4-RECOVER] $1" | tee -a orchestrate.log; }

# 1. 0.5B graph eval (training already done)
log "Evaluating 0.5B graph on GPU 4..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct \
    --lora_path experiments/scale_0.5B_graph/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_0.5B_graph/eval_merged_rerank \
    --gpu_id 4 --top_k 200 --max_seq_length 512 \
    2>&1 | tee experiments/scale_0.5B_graph_eval_v2.log
log "0.5B graph eval done."

# 2. 0.5B bm25 re-training (clean up failed run first)
rm -rf experiments/scale_0.5B_bm25only
log "Re-training 0.5B bm25only on GPU 4..."
$PYTHON src/train/train_rankft.py \
    --model_path /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/scale_0.5B_bm25only \
    --device cuda:4 --num_negatives 16 \
    --neg_bm25_ratio 1.0 --neg_graph_ratio 0.0 --neg_random_ratio 0.0 \
    --learning_rate 5e-5 --num_epochs 2 --batch_size 1 \
    --gradient_accumulation_steps 16 --save_steps 800 --logging_steps 10 \
    --max_seq_length 512 --lora_rank 16 --seed 42 \
    2>&1 | tee experiments/scale_0.5B_bm25only.log
log "0.5B bm25 training done."

# 3. 0.5B bm25 eval
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/kangshijia/models/huggingface/Qwen2.5-0.5B-Instruct \
    --lora_path experiments/scale_0.5B_bm25only/final \
    --test_data data/grepo_text/grepo_test.jsonl \
    --bm25_candidates data/rankft/merged_bm25_exp6_candidates.jsonl \
    --output_dir experiments/scale_0.5B_bm25only/eval_merged_rerank \
    --gpu_id 4 --top_k 200 --max_seq_length 512 \
    2>&1 | tee experiments/scale_0.5B_bm25only_eval.log
log "0.5B bm25 eval done."

# 4. BeetleBox Java eval
log "Starting BeetleBox Java eval on GPU 4..."
$PYTHON src/eval/eval_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/rankft_runB_graph/best \
    --test_data /data/chenlibin/beetlebox/java_test.jsonl \
    --bm25_candidates /data/chenlibin/beetlebox/java_bm25_top500.jsonl \
    --output_dir experiments/beetlebox_java_eval \
    --gpu_id 4 --top_k 200 --max_seq_length 512 \
    2>&1 | tee experiments/beetlebox_java_eval.log
log "BeetleBox done."

# 5. 14B scale
log "Starting 14B scale on GPU 4..."
bash scripts/launch_scale_ablation.sh 14B 4 2>&1 | tee experiments/scale_14B.log
log "14B scale done."

log "GPU4 recovery COMPLETE."
