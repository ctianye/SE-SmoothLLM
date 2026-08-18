#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/se-smoothllm/repo}"
MODEL_DIR="${MODEL_DIR:-/root/se-smoothllm/models/Meta-Llama-3-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-meta-llama/Llama-3-8b-chat-hf}"

export HF_HOME="${HF_HOME:-/root/se-smoothllm/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/se-smoothllm/modelscope}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/se-smoothllm/cache}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "未找到 Llama-3-8B 模型配置：${MODEL_DIR}/config.json" >&2
  exit 1
fi
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "未找到项目目录：${PROJECT_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --generation-config vllm
