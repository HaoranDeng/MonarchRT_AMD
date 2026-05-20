#!/bin/bash

#SBATCH -J QKV_Attn_Exp       # Job name
#SBATCH -o out/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH -e out/job.%j.err     # Name of stderr error file
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 1:00:00            # Run time (hh:mm:ss) - 1 hour
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

python experiments/bm41_qkv/analyze_qkv_attention.py \
  --npz assets/first_qkv/first_attn_qkv_dense_layer0_ts999.npz \
  --device cuda \
  --block-size 4 \
  --quick-tokens 4096 \
  --monarch-iters 1 10 \
  --attn-slice 0,0
