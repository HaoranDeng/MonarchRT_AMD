#!/bin/bash

#SBATCH -J BM41_Compare       # Job name
#SBATCH -o out/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH -e out/job.%j.err     # Name of stderr error file
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 1:00:00            # Run time (hh:mm:ss) - 1 hour
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

NPZ="${NPZ:-}"
DEVICE="${DEVICE:-cuda}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
QUICK_TOKENS="${QUICK_TOKENS:-4096}"
BATCH="${BATCH:-2}"
HEADS="${HEADS:-4}"
TOKENS="${TOKENS:-512}"
DIM="${DIM:-64}"
CHECK_ATTN_MATRIX="${CHECK_ATTN_MATRIX:-0}"

args=(
  --device "$DEVICE"
  --block-size "$BLOCK_SIZE"
  --quick-tokens "$QUICK_TOKENS"
)

if [[ -n "$NPZ" ]]; then
  args+=(--npz "$NPZ")
else
  args+=(--batch "$BATCH" --heads "$HEADS" --tokens "$TOKENS" --dim "$DIM")
fi

if [[ "$CHECK_ATTN_MATRIX" == "1" ]]; then
  args+=(--check-attn-matrix)
fi

python scripts/compare_bm41_attention.py "${args[@]}"
