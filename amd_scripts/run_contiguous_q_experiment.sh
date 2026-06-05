#!/bin/bash

#SBATCH -J BM41_ContigQ
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00
#SBATCH -p mi3001x

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

export BM41_LAYOUT=contiguous_q
export BM41_Q_INIT=ith
export BM41_RANDOM_SEED="${BM41_RANDOM_SEED:-0}"

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-contiguous-${SLURM_JOB_ID:-manual}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

OUTPUT_FOLDER="videos/contiguous_ith"

mkdir -p "${OUTPUT_FOLDER}"
find "${OUTPUT_FOLDER}" -mindepth 1 -delete

echo "Running BM41 layout=${BM41_LAYOUT}, q_init=${BM41_Q_INIT}, output=${OUTPUT_FOLDER}"
python inference.py \
  --config_path configs/wan_bm41_fewstep_dmd.yaml \
  --output_folder "${OUTPUT_FOLDER}" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --max_prompts 10 \
  --use_ema
