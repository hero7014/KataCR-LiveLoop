#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${KATACR_RUNTIME_ROOT:-$SCRIPT_ROOT}"
SOURCE_ROOT="${KATACR_SOURCE_ROOT:-$RUNTIME_ROOT}"
VENV_ROOT="${KATACR_VENV_ROOT:-$RUNTIME_ROOT/.venv}"
RUNTIME_SITE="${KATACR_RUNTIME_SITE:-$VENV_ROOT/lib/python3.11/site-packages}"
PYTHON_BIN="${KATACR_PYTHON_BIN:-$VENV_ROOT/bin/python}"
NVIDIA_LIB_ROOT="${KATACR_NVIDIA_LIB_ROOT:-$VENV_ROOT/lib/python3.11/site-packages/nvidia}"

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$NVIDIA_LIB_ROOT/cudnn/lib:$NVIDIA_LIB_ROOT/cublas/lib:$NVIDIA_LIB_ROOT/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$RUNTIME_SITE:$RUNTIME_ROOT:$SOURCE_ROOT/.venv/lib/python3.11/site-packages:${PYTHONPATH:-}"

cd "$RUNTIME_ROOT"
exec "$PYTHON_BIN" -u scripts/adb_shadow_visual.py "$@"
