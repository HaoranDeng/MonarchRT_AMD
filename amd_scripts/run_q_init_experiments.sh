#!/bin/bash

#SBATCH -J BM41_QInit
#SBATCH -o out/job.%A_%a.out
#SBATCH -e out/job.%A_%a.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00
#SBATCH -p mi3001x
#SBATCH --array=0-3

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

Q_INITS=(mean random 1st ith)
Q_INIT="${Q_INITS[${SLURM_ARRAY_TASK_ID:-0}]}"

export BM41_Q_INIT="${Q_INIT}"
export BM41_RANDOM_SEED="${BM41_RANDOM_SEED:-0}"

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

OUTPUT_FOLDER="videos/${Q_INIT}"

mkdir -p "${OUTPUT_FOLDER}"
find "${OUTPUT_FOLDER}" -mindepth 1 -delete

echo "Running BM41 q_init=${Q_INIT}, output=${OUTPUT_FOLDER}"
python inference.py \
  --config_path configs/wan_bm41_fewstep_dmd.yaml \
  --output_folder "${OUTPUT_FOLDER}" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --max_prompts 10 \
  --use_ema
