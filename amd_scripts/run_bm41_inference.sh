#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${CONFIG_PATH:-configs/wan_bm41_fewstep_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DATA_PATH="${DATA_PATH:-prompts/MovieGenVideoBench.txt}"
EXTENDED_PROMPT_PATH="${EXTENDED_PROMPT_PATH:-}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-videos_bm41}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --config_path "$CONFIG_PATH"
  --data_path "$DATA_PATH"
  --output_folder "$OUTPUT_FOLDER"
  --max_prompts "$MAX_PROMPTS"
  --num_samples "$NUM_SAMPLES"
  --seed "$SEED"
  --save_with_index
)

if [[ -n "$CHECKPOINT_PATH" ]]; then
  args+=(--checkpoint_path "$CHECKPOINT_PATH")
fi

if [[ -n "$EXTENDED_PROMPT_PATH" ]]; then
  args+=(--extended_prompt_path "$EXTENDED_PROMPT_PATH")
fi

"$PYTHON_BIN" inference.py "${args[@]}"
