#!/bin/bash

#SBATCH -J MonarchRT_INF
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00
#SBATCH -p mi2101x

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

OUTPUT_FOLDER="videos/monarch"

mkdir -p "${OUTPUT_FOLDER}"
find "${OUTPUT_FOLDER}" -mindepth 1 -delete

python inference.py \
  --config_path configs/self_forcing_monarch_dmd.yaml \
  --output_folder "${OUTPUT_FOLDER}" \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --use_ema
