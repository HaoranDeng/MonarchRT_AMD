#!/bin/bash

#SBATCH -J Save_QKV           # Job name
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

export MONARCHRT_SAVE_FIRST_QKV_PATH=assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz

python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder videos/qkv_capture \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --max_prompts 1 \
  --use_ema
