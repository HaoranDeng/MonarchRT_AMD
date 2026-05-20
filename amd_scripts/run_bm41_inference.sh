#!/bin/bash

#SBATCH -J BM41_Infer         # Job name
#SBATCH -o out/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH -e out/job.%j.err     # Name of stderr error file
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 4:00:00            # Run time (hh:mm:ss) - 4 hours
#SBATCH -p mi3001x            # Desired partition

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

BM41_BLOCK_SIZE="${BM41_BLOCK_SIZE:-4}"

python inference.py \
  --config_path configs/wan_bm41_fewstep_dmd.yaml \
  --output_folder videos/bm41 \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --max_prompts 10 \
  --bm41_block_size "${BM41_BLOCK_SIZE}" \
  --use_ema
