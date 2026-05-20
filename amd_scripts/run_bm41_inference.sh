#!/bin/bash

#SBATCH -J BM41_Infer         # Job name
#SBATCH -o out/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH -e out/job.%j.err     # Name of stderr error file
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 4:00:00            # Run time (hh:mm:ss) - 4 hours
#SBATCH -p mi3001x            # Desired partition

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${WORK:-}" ] && [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${WORK}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate "${CONDA_ENV:-monarch_rt}"

mkdir -p out

CONFIG_PATH="${CONFIG_PATH:-configs/wan_bm41_fewstep_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/self_forcing_dmd.pt}"
DATA_PATH="${DATA_PATH:-prompts/MovieGenVideoBench_extended.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-videos/bm41}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"

args=(
  --config_path "$CONFIG_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --data_path "$DATA_PATH"
  --output_folder "$OUTPUT_FOLDER"
  --max_prompts "$MAX_PROMPTS"
  --use_ema
)

if [ -n "${EXTENDED_PROMPT_PATH:-}" ]; then
  args+=(--extended_prompt_path "$EXTENDED_PROMPT_PATH")
fi

if [ -n "${NUM_SAMPLES:-}" ]; then
  args+=(--num_samples "$NUM_SAMPLES")
fi

if [ -n "${SEED:-}" ]; then
  args+=(--seed "$SEED")
fi

if [ "${SAVE_WITH_INDEX:-0}" = "1" ]; then
  args+=(--save_with_index)
fi

python inference.py "${args[@]}"
