#!/bin/bash
# Auto-launch: monitors GPU availability and launches next priority experiment
# Usage: nohup bash scripts/auto_launch.sh &
#
# Priority queue:
# 1. Eval exp9/10/11 (when training finishes)
# 2. RankFT Run C (random negatives control)
# 3. RankFT Run E (content-aware, fresh LoRA)
# 4. Eval all completed RankFT runs
# 5. Zero-shot reranker baseline

cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python"

log() { echo "[$(date +%H:%M:%S)] $*"; }

check_gpu_free() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$mem_used" ]; then return 1; fi
    # Consider free if < 1000 MiB used
    if [ "$mem_used" -lt 1000 ]; then
        return 0
    fi
    return 1
}

find_free_gpu() {
    # Check GPUs 0-7, prefer 4-7 first
    for gpu_id in 4 5 6 7 0 1 2 3; do
        if check_gpu_free $gpu_id; then
            echo $gpu_id
            return 0
        fi
    done
    return 1
}

# Track what we've launched
LAUNCHED=""

while true; do
    FREE_GPU=$(find_free_gpu)
    if [ -z "$FREE_GPU" ]; then
        sleep 120
        continue
    fi

    log "GPU $FREE_GPU is free!"

    # Priority 1: Eval completed SFT experiments
    for exp_dir in exp9_tgs_filetree exp10_tgs_graph exp11_navcot; do
        final_path="experiments/${exp_dir}/stage2_sft/final"
        eval_done="experiments/${exp_dir}/eval_filetree/summary.json"

        # Check if training is done (final checkpoint exists) and eval not done
        if [ -d "$final_path" ] && [ ! -f "$eval_done" ]; then
            if echo "$LAUNCHED" | grep -q "${exp_dir}_eval"; then continue; fi
            log "Launching eval for ${exp_dir} on GPU $FREE_GPU"
            CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONUNBUFFERED=1 \
            $PYTHON src/eval/eval_grepo_file_level.py \
                --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
                --lora_path "$final_path" \
                --test_data data/grepo_text/grepo_test.jsonl \
                --file_tree_dir data/file_trees \
                --output_dir "experiments/${exp_dir}/eval_filetree" \
                --prompt_mode filetree &
            LAUNCHED="$LAUNCHED ${exp_dir}_eval"
            sleep 30  # Let it start
            break
        fi
    done

    # Re-check if GPU was taken
    if ! check_gpu_free $FREE_GPU; then
        sleep 120
        continue
    fi

    # Priority 2: RankFT Run C (random negatives control)
    if [ ! -d "experiments/rankft_runC_random/final" ] && \
       ! echo "$LAUNCHED" | grep -q "rankft_runC"; then
        log "Launching RankFT Run C on GPU $FREE_GPU"
        nohup bash experiments/rankft_runC_random/launch.sh $FREE_GPU \
            > experiments/rankft_runC_random/train.log 2>&1 &
        LAUNCHED="$LAUNCHED rankft_runC"
        sleep 120
        continue
    fi

    # Priority 3: RankFT Run E (content-aware, fresh LoRA)
    if [ -f "data/file_summaries_aligned.json" ] && \
       [ ! -d "experiments/rankft_runE_content_fresh/final" ] && \
       ! echo "$LAUNCHED" | grep -q "rankft_runE"; then
        log "Launching RankFT Run E (content fresh) on GPU $FREE_GPU"
        nohup bash experiments/rankft_runE_content_fresh/launch.sh $FREE_GPU \
            > experiments/rankft_runE_content_fresh/train.log 2>&1 &
        LAUNCHED="$LAUNCHED rankft_runE"
        sleep 120
        continue
    fi

    # Priority 4: Eval completed RankFT runs
    for run_label in rankft_runA_bm25only rankft_runB_graph rankft_runC_random rankft_runD_content rankft_runE_content_fresh; do
        final_path="experiments/${run_label}/final"
        eval_grepo="experiments/${run_label}_grepo_k200/summary.json"

        if [ -d "$final_path" ] && [ ! -f "$eval_grepo" ]; then
            if echo "$LAUNCHED" | grep -q "${run_label}_eval"; then continue; fi
            log "Launching RankFT eval for ${run_label} on GPU $FREE_GPU"

            # Determine if content-aware eval needed
            if echo "$run_label" | grep -q "content"; then
                EVAL_SCRIPT="src/eval/eval_rankft_content.py"
            else
                EVAL_SCRIPT="src/eval/eval_rankft.py"
            fi

            # Determine summary file per dataset
            if echo "$run_label" | grep -q "content"; then
                GREPO_SUMMARY_ARGS="--file_summaries data/file_summaries_aligned.json"
                SWEBENCH_SUMMARY_ARGS="--file_summaries data/swebench_file_summaries/file_summaries_all.json"
            else
                GREPO_SUMMARY_ARGS=""
                SWEBENCH_SUMMARY_ARGS=""
            fi

            # GREPO eval, then SWE-bench eval
            CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONUNBUFFERED=1 \
            $PYTHON $EVAL_SCRIPT \
                --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
                --lora_path "$final_path" \
                --test_data data/grepo_text/grepo_test.jsonl \
                --bm25_candidates data/rankft/grepo_test_bm25_top500.jsonl \
                --output_dir "experiments/${run_label}_grepo_k200" \
                --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16 \
                $GREPO_SUMMARY_ARGS && \
            CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONUNBUFFERED=1 \
            $PYTHON $EVAL_SCRIPT \
                --model_path /data/shuyang/models/Qwen2.5-7B-Instruct \
                --lora_path "$final_path" \
                --test_data data/swebench_lite/swebench_lite_test.jsonl \
                --bm25_candidates data/rankft/swebench_test_bm25_top500.jsonl \
                --output_dir "experiments/${run_label}_swebench_k200" \
                --gpu_id 0 --top_k 200 --max_seq_length 512 --score_batch_size 16 \
                $SWEBENCH_SUMMARY_ARGS &

            LAUNCHED="$LAUNCHED ${run_label}_eval"
            sleep 30
            break
        fi
    done

    # Re-check if GPU was taken
    if ! check_gpu_free $FREE_GPU; then
        sleep 120
        continue
    fi

    # Priority 5: Zero-shot reranker baseline
    if [ ! -f "experiments/zeroshot_rankft/summary.json" ] && \
       ! echo "$LAUNCHED" | grep -q "zeroshot"; then
        log "Launching zero-shot reranker eval on GPU $FREE_GPU"
        nohup bash scripts/eval_zeroshot_reranker.sh $FREE_GPU \
            "experiments/exp1_sft_only/stage2_sft/final" \
            > experiments/zeroshot_eval.log 2>&1 &
        LAUNCHED="$LAUNCHED zeroshot"
        sleep 120
        continue
    fi

    log "All priority tasks launched or completed."
    sleep 300
done
