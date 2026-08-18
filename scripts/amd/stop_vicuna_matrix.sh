#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
PID_FILE="${STATE_DIR}/vicuna-matrix.pid"

source "${SCRIPT_DIR}/vicuna_matrix_common.sh"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "未找到 Vicuna 生成矩阵 PID 文件。"
  exit 0
fi

matrix_pid="$(cat "${PID_FILE}")"
if ! matrix_process_is_running "${matrix_pid}"; then
  echo "Vicuna 生成矩阵已经停止。"
  rm -f "${PID_FILE}"
  exit 0
fi

kill -TERM -- "-${matrix_pid}" 2>/dev/null || kill -TERM "${matrix_pid}"
echo "已请求停止 Vicuna 生成矩阵，PID=${matrix_pid}。已写入的 JSONL 可继续使用。"
