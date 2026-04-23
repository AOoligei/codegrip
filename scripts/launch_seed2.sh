#!/bin/bash
# Launch seed=2 robustness runs on GPU 1 (free after structural ablation evals finish ~17:40)
# Run: bash scripts/launch_seed2.sh
set -e
BASE=/home/chenlibin/grepo_agent
cd $BASE

GPU=1  # change if GPU 1 is not free

echo "[$(date)] Starting rankft_runB_graph_seed2 on GPU $GPU" | tee -a seed_robustness.log

nohup bash -c "
conda run -n tgn python src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/rankft_runB_graph_seed2 \
    --device cuda:$GPU \
    --num_negatives 16 \
    --neg_bm25_ratio 0.5 \
    --neg_graph_ratio 0.25 \
    --neg_random_ratio 0.25 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 800 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank 32 \
    --seed 2 \
    >> experiments/rankft_runB_graph_seed2.log 2>&1

echo \"[$(date)] Finished seed2 graph. Starting seed2 bm25only\" >> seed_robustness.log

conda run -n tgn python src/train/train_rankft.py \
    --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
    --lora_path experiments/exp1_sft_only/stage2_sft/final \
    --train_data data/grepo_text/grepo_train.jsonl \
    --bm25_candidates data/rankft/grepo_train_bm25_top500.jsonl \
    --dep_graph_dir data/dep_graphs \
    --train_data_for_cochange data/grepo_text/grepo_train.jsonl \
    --file_tree_dir data/file_trees \
    --output_dir experiments/rankft_runA_bm25only_seed2 \
    --device cuda:$GPU \
    --num_negatives 16 \
    --neg_bm25_ratio 1.0 \
    --neg_graph_ratio 0.0 \
    --neg_random_ratio 0.0 \
    --learning_rate 5e-5 \
    --num_epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 800 \
    --logging_steps 10 \
    --max_seq_length 512 \
    --lora_rank 32 \
    --seed 2 \
    >> experiments/rankft_runA_bm25only_seed2.log 2>&1

echo \"[$(date)] Finished both seed=2 runs\" >> seed_robustness.log
" > /dev/null 2>&1 &

echo "Seed=2 chain running in background (PID: $!)"
echo "Logs: experiments/rankft_runB_graph_seed2.log, experiments/rankft_runA_bm25only_seed2.log"
