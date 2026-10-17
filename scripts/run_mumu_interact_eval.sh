#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KATACR_RUNTIME_ROOT:-$SCRIPT_ROOT}"
SOURCE_ROOT="${KATACR_SOURCE_ROOT:-$RUNTIME_ROOT}"
VENV_ROOT="${KATACR_VENV_ROOT:-$RUNTIME_ROOT/.venv}"
RUNTIME_SITE="${KATACR_RUNTIME_SITE:-$VENV_ROOT/lib/python3.11/site-packages}"
PYTHON_BIN="${KATACR_PYTHON_BIN:-$VENV_ROOT/bin/python}"
NVIDIA_LIB_ROOT="${KATACR_NVIDIA_LIB_ROOT:-$VENV_ROOT/lib/python3.11/site-packages/nvidia}"

export KATACR_INTERACT_SOURCE="${KATACR_INTERACT_SOURCE:-adb}"
export KATACR_MUMU_SIMPLE_START="${KATACR_MUMU_SIMPLE_START:-1}"
export KATACR_ADB="${KATACR_ADB:-adb}"
export KATACR_ADB_SERIAL="${KATACR_ADB_SERIAL:-127.0.0.1:16384}"
export KATACR_ADB_NORMALIZE="${KATACR_ADB_NORMALIZE:-resize_width}"
export KATACR_TIME_WALLCLOCK_FALLBACK="${KATACR_TIME_WALLCLOCK_FALLBACK:-1}"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$NVIDIA_LIB_ROOT/cudnn/lib:$NVIDIA_LIB_ROOT/cublas/lib:$NVIDIA_LIB_ROOT/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$RUNTIME_SITE:$RUNTIME_ROOT:$SOURCE_ROOT/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

"$KATACR_ADB" connect "$KATACR_ADB_SERIAL" >/dev/null 2>&1 || true

cd "$RUNTIME_ROOT"
exec "$PYTHON_BIN" -u katacr/policy/offline/eval_no_delay.py \
  --eval-num "${KATACR_EVAL_NUM:-1}" \
  --show "${KATACR_SHOW:-False}" \
  --save "${KATACR_SAVE:-True}" \
  --verbose "${KATACR_VERBOSE:-True}" \
  --show-predict "${KATACR_SHOW_PREDICT:-True}" \
  --save-frame-log "${KATACR_SAVE_FRAME_LOG:-True}" \
  "$@"
