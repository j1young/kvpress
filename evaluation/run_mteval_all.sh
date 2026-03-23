#!/usr/bin/env bash
# Run mteval_kvzip.py for all (dataset, compression_ratio, press) combinations.
# GPU 0~3 each handles one compression_ratio (0.5, 0.7, 0.8, 0.9) in parallel.
# Within each GPU, (dataset x press) jobs run sequentially.

set -euo pipefail

MODEL="/remote/vast0/share/model/Qwen3-8B"
SCRIPT="evaluation/mteval_kvzip.py"
LOG_DIR="./results/mteval/logs"
mkdir -p "$LOG_DIR"

DATASETS=(
    "evaluation/merged_expansion_multi_generated.json"
    "evaluation/merged_refinement_multi_generated.json"
    "evaluation/merged_follow-up_multi.json"
)
COMPRESSION_RATIOS=(0.5 0.7 0.8 0.9)
PRESSES=(kvzip fastkvzip expected_attention)

# Each GPU runs all (dataset x press) combinations for its assigned compression_ratio
run_gpu() {
    local gpu_id=$1
    local cr=$2
    local fail=0

    for dataset in "${DATASETS[@]}"; do
        for press in "${PRESSES[@]}"; do
            local tag
            tag="$(basename "$dataset" .json)_${press}_cr${cr}"
            local logfile="${LOG_DIR}/${tag}.log"

            echo "[GPU ${gpu_id}] START  ${tag}"
            if CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SCRIPT}" \
                --model "${MODEL}" \
                --dataset "${dataset}" \
                --compression_ratio "${cr}" \
                --press "${press}" \
                > "${logfile}" 2>&1; then
                echo "[GPU ${gpu_id}] DONE   ${tag}"
            else
                echo "[GPU ${gpu_id}] FAIL   ${tag} (exit=$?), see ${logfile}"
                ((fail++))
            fi
        done
    done
    return $fail
}

echo "Datasets: ${#DATASETS[@]}, Presses: ${#PRESSES[@]}, Ratios: ${#COMPRESSION_RATIOS[@]}"
echo "Total jobs: $(( ${#DATASETS[@]} * ${#PRESSES[@]} * ${#COMPRESSION_RATIOS[@]} ))"
echo "GPU assignment: 0->0.5, 1->0.7, 2->0.8, 3->0.9"
echo "=========================================="

PIDS=()
for i in "${!COMPRESSION_RATIOS[@]}"; do
    run_gpu "$i" "${COMPRESSION_RATIOS[$i]}" &
    PIDS+=($!)
done

FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAIL_COUNT++))
done

echo "=========================================="
echo "All done. GPUs with failures: ${FAIL_COUNT}/4"
echo "Logs: ${LOG_DIR}/"
