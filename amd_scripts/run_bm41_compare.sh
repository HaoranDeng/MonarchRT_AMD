#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPZ="${NPZ:-}"
DEVICE="${DEVICE:-cuda}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
QUICK_TOKENS="${QUICK_TOKENS:-4096}"
BATCH="${BATCH:-2}"
HEADS="${HEADS:-4}"
TOKENS="${TOKENS:-512}"
DIM="${DIM:-64}"
CHECK_ATTN_MATRIX="${CHECK_ATTN_MATRIX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

"$PYTHON_BIN" scripts/compare_bm41_attention.py "${args[@]}"
