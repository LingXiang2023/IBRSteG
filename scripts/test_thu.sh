#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER=( "$PYTHON_BIN" test_thu.py )
if [[ -d "${RASTERIZER_BUILD:-.local_build/diff_gaussian_rasterization}/diff_gaussian_rasterization" ]]; then
  RUNNER=( "$PYTHON_BIN" tools/run_with_local_rasterizer.py --rasterizer-build "${RASTERIZER_BUILD:-.local_build/diff_gaussian_rasterization}" test_thu.py -- )
fi

"${RUNNER[@]}" \
  --data-root "${THU_DATA_ROOT:-data/thu_processed}" \
  --gps-checkpoint "${GPS_CHECKPOINT:-model_zoo/gps_plus_final.pth}" \
  --gas-checkpoint "${GAS_CHECKPOINT:-model_zoo/ibrsteg_test_weight.pth}" \
  --output-dir "${OUTPUT_DIR:-results/thu_test}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@"
