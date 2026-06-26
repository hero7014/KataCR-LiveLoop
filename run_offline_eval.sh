#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment: $(pwd)/.venv"
  exit 1
fi

VIDEO_PATH="${1:-/mnt/c/Users/feng/Downloads/Screenrecording_20250714_155704_com.tencent.tmgp..mp4}"
MODEL_NAME="StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252"
LOAD_EPOCH="1"

echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_NAME"
echo "Epoch: $LOAD_EPOCH"

"$PYTHON_BIN" katacr/policy/offline/eval_no_delay.py \
  --model-name "$MODEL_NAME" \
  --load-epoch "$LOAD_EPOCH" \
  --vid-path "$VIDEO_PATH" \
  --eval-num 1 \
  --show false \
  --save true \
  --deterministic true \
  --verbose true
