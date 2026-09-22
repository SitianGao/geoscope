#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${1:-RemoteAgent}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-RemoteAgent}"
PORT="${2:-8000}"
GPU_ID="${GPU_ID:-0}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
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
echo "Service address: $BIND_HOST:$PORT"
echo "GPU device: $GPU_ID"
echo "Starting vLLM service..."

unset PYTHONPATH
exec env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONNOUSERSITE=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$BIND_HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":1048576}' \
  --enforce-eager
