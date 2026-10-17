#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment: $(pwd)/.venv"
  exit 1
fi

NVIDIA_LIB_ROOT="$(pwd)/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$NVIDIA_LIB_ROOT/cudnn/lib:$NVIDIA_LIB_ROOT/cublas/lib:$NVIDIA_LIB_ROOT/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

VIDEO_PATH="${1:-${KATACR_VIDEO_PATH:-}}"
if [[ -z "$VIDEO_PATH" ]]; then
  echo "Usage: $0 /path/to/match.mp4" >&2
  echo "Or set KATACR_VIDEO_PATH." >&2
  exit 2
fi
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
  --verbose true \
  --show-predict false \
  --fixed-window true \
  --reward-ocr-interval 10 \
  --time-ocr-interval 3 \
  --elixir-ocr-interval 2 \
  --save-video false \
  --save-video-interval 10
