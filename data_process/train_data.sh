#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${THU_RAW_ROOT:-data/thu_raw}"
PROCESSED_ROOT="${THU_PROCESSED_ROOT:-data/thu_processed}"

TRAIN_SEQS="${TRAIN_SEQS-s1a1 s1a2 s1a3 s2a1 s2a2 s2a3 s3a1 s3a2 s3a3}"
VAL_SEQS="${VAL_SEQS-s3a5}"
TEST_SEQS="${TEST_SEQS-s1a4 s1a5 s1a6 s2a4 s3a5}"

run_seq() {
  local seq="$1"
  local split="$2"
  local extra_args=()
  if [[ -n "${MAX_FRAMES:-}" ]]; then
    extra_args+=(--max-frames "$MAX_FRAMES")
  fi
  "$PYTHON_BIN" data_process/step_0rect.py \
    -i "$seq" \
    -t "$split" \
    --data-root "$DATA_ROOT" \
    --processed-data-root "$PROCESSED_ROOT" \
    "${extra_args[@]}"
  "$PYTHON_BIN" data_process/step_1.py \
    -i "$seq" \
    -t "$split" \
    --data-root "$DATA_ROOT" \
    --processed-data-root "$PROCESSED_ROOT" \
    "${extra_args[@]}"
}

for seq in $TRAIN_SEQS; do
  run_seq "$seq" train
done

for seq in $VAL_SEQS; do
  run_seq "$seq" val
done

for seq in $TEST_SEQS; do
  run_seq "$seq" test
done
