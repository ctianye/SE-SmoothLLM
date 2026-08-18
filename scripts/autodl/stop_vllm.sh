#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/vllm.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "未找到 ${PID_FILE}，无需停止。"
  exit 0
fi

VLLM_PID="$(cat "${PID_FILE}")"
if [[ ! "${VLLM_PID}" =~ ^[0-9]+$ ]]; then
  echo "PID 文件内容无效：${PID_FILE}" >&2
  exit 1
fi
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "进程已不存在，已清理过期 PID 文件。"
  exit 0
fi

kill "${VLLM_PID}"
for _ in {1..30}; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "vLLM 已停止。"
    exit 0
  fi
  sleep 1
done

echo "vLLM 在 30 秒内未退出，请人工检查 PID=${VLLM_PID}。" >&2
exit 1
