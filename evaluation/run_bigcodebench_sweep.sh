#!/usr/bin/env bash
# Sweep bigcodebench_kvzip.py over multiple press algorithms, compression ratios,
# and enable_thinking on/off.

set -euo pipefail

MODEL="/remote/vast0/share/model/Qwen3-8B"
RATIOS=(0.5 0.6 0.7 0.8 0.9)
PRESSES=("kvzip" "fastkvzip" "expected_attention" "kvzap")
THINKING_FLAGS=("--no-enable_thinking" "--enable_thinking")

python evaluation/bigcodebench_kvzip.py \
  --model "$MODEL" \
  --no-enable_thinking

for thinking in "${THINKING_FLAGS[@]}"; do
  for press in "${PRESSES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      echo "=========================================="
      echo "  ${thinking}  press=${press}  compression_ratio=${ratio}"
      echo "=========================================="
      python evaluation/bigcodebench_kvzip.py \
        --model "$MODEL" \
        --press "$press" \
        --compression_ratio "$ratio" \
        "$thinking"
    done
  done
done

python evaluation/bigcodebench_kvzip.py \
  --model "$MODEL" \
  --enable_thinking

echo "All runs complete."
