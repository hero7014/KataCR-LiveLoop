#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
PIP_BIN=".venv/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment: $(pwd)/.venv"
  echo "Create it first, then rerun this script."
  exit 1
fi

echo "[1/3] Removing conflicting JAX stack packages..."
"$PIP_BIN" uninstall -y \
  jax \
  jaxlib \
  flax \
  optax \
  orbax-checkpoint \
  chex \
  tensorstore \
  ml-dtypes

echo "[2/3] Reinstalling pinned compatible versions..."
"$PIP_BIN" install \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "jax==0.4.26" \
  "jaxlib==0.4.26" \
  "flax==0.8.1" \
  "optax==0.2.2" \
  "orbax-checkpoint==0.5.3" \
  "chex==0.1.86" \
  "tensorstore==0.1.84" \
  "ml-dtypes==0.5.4"

echo "[3/3] Running import smoke test..."
"$PYTHON_BIN" scripts/check_runtime.py

echo "Environment repair completed."
