#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${1:-RemoteAgent}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-RemoteAgent}"
PORT="${2:-8000}"
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${ROOT_DIR}/${MODEL_DIR}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Error: model directory does not exist: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Error: config.json was not found under: $MODEL_PATH"
  exit 1
fi

echo "Model path: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Service port: $PORT"
echo "GPU device: $GPU_ID"
echo "Starting vLLM service..."

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":2,"video":2}'
