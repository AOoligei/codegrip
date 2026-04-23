#!/bin/bash
set -euo pipefail
cd /home/chenlibin/grepo_agent
PYTHON="/home/chenlibin/miniconda3/envs/tgn/bin/python3"
MODEL_7B="/data/shuyang/models/Qwen2.5-7B-Instruct"
MODEL_3B="/data/chenlibin/models/Qwen2.5-3B-Instruct"
TRICKED="data/rankft/swebench_bm25_tricked_top500.jsonl"
EVAL_GPU=7  # All eval on GPU 7 (shared with 3B training ~12G)

declare -A EXPS
EXPS[swebench_adapted_delex50]="$MODEL_7B"
EXPS[swebench_adapted_5ep]="$MODEL_7B"
EXPS[swebench_adapted_lr2e5]="$MODEL_7B"
EXPS[swebench_adapted_3B]="$MODEL_3B"
EXPS[swebench_adapted_delex50_5ep]="$MODEL_7B"
EXPS[swebench_adapted_delex50_lr2e5]="$MODEL_7B"
EXPS[swebench_adapted_3B_delex50]="$MODEL_3B"
EXPS[swebench_adapted_5ep_lr2e5]="$MODEL_7B"

declare -A DONE

while true; do
    for exp in "${!EXPS[@]}"; do
        [[ -n "${DONE[$exp]:-}" ]] && continue
        DIR="experiments/${exp}"
        if [ ! -f "${DIR}/final/adapter_config.json" ]; then
            if [ -f "${DIR}/training_diagnostics.jsonl" ]; then
                STEP=$(tail -1 "${DIR}/training_diagnostics.jsonl" 2>/dev/null | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('step','?'))" 2>/dev/null || echo "?")
                echo "[$(date)] ${exp}: step ${STEP}"
            fi
            continue
        fi

        echo "[$(date)] ${exp} DONE! Batch eval on GPU ${EVAL_GPU}..."
        bash scripts/batch_eval_checkpoints.sh "${DIR}" "${EXPS[$exp]}" "${EVAL_GPU}" "${TRICKED}"
        
        if [ -f "${DIR}/best_checkpoint_result.json" ]; then
            $PYTHON -c "
import json
d = json.load(open('${DIR}/best_checkpoint_result.json'))
print(f'>>> ${exp}: BM25={d[\"best_bm25_h1\"]:.2f}% ({d[\"best_bm25_ckpt\"]}) Tricked={d[\"best_tricked_h1\"]:.2f}% ({d[\"best_tricked_ckpt\"]})')
"
            DONE[$exp]=1
        fi
    done

    ALL_DONE=1
    for exp in "${!EXPS[@]}"; do
        [[ -z "${DONE[$exp]:-}" ]] && ALL_DONE=0 && break
    done

    if [ "${ALL_DONE}" -eq 1 ]; then
        echo ""
        echo "========== ALL RESULTS =========="
        printf "%-35s %10s %12s %10s %12s\n" "Experiment" "BM25" "ckpt" "Tricked" "ckpt"
        printf "%-35s %10s %12s %10s %12s\n" "Baseline" "50.67%" "best" "53.67%" "best"
        for exp in swebench_adapted_delex50 swebench_adapted_5ep swebench_adapted_lr2e5 swebench_adapted_3B swebench_adapted_delex50_5ep swebench_adapted_delex50_lr2e5 swebench_adapted_3B_delex50 swebench_adapted_5ep_lr2e5; do
            DIR="experiments/${exp}"
            if [ -f "${DIR}/best_checkpoint_result.json" ]; then
                $PYTHON -c "
import json
d = json.load(open('${DIR}/best_checkpoint_result.json'))
print(f\"%-35s %10s %12s %10s %12s\" % ('${exp}', f\"{d['best_bm25_h1']:.2f}%\", d['best_bm25_ckpt'], f\"{d['best_tricked_h1']:.2f}%\", d['best_tricked_ckpt']))
"
            else
                printf "%-35s %10s %12s %10s %12s\n" "${exp}" "N/A" "-" "N/A" "-"
            fi
        done
        break
    fi
    sleep 120
done
