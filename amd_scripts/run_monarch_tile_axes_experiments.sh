#!/bin/bash

#SBATCH -J Monarch_TileAxes
#SBATCH -o out/job.%A_%a.out
#SBATCH -e out/job.%A_%a.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -p mi2101x
#SBATCH --array=0-1

set -euo pipefail

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

conda activate monarch_rt

TILE_AXES_LIST=(row_row col_col)
TILE_AXES="${TILE_AXES_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

export MONARCH_TILE_AXES="${TILE_AXES}"
export MONARCH_Q_INIT="${MONARCH_Q_INIT:-mean}"
export MONARCH_RANDOM_SEED="${MONARCH_RANDOM_SEED:-0}"

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

OUTPUT_FOLDER="videos/${TILE_AXES}"

mkdir -p "${OUTPUT_FOLDER}"
find "${OUTPUT_FOLDER}" -mindepth 1 -delete

echo "Running Monarch tile_axes=${TILE_AXES}, q_init=${MONARCH_Q_INIT}, output=${OUTPUT_FOLDER}"
python inference.py \
  --config_path configs/wan_monarch_fewstep_dmd.yaml \
  --output_folder "${OUTPUT_FOLDER}" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --max_prompts 10 \
  --use_ema
