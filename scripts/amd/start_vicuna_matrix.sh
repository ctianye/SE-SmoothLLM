#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
PID_FILE="${STATE_DIR}/vicuna-matrix.pid"
MASTER_LOG="${LOG_DIR}/vicuna-matrix.log"

source "${SCRIPT_DIR}/vicuna_matrix_common.sh"

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if matrix_process_is_running "${existing_pid}"; then
    echo "Vicuna 生成矩阵已在运行，PID=${existing_pid}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

command -v setsid >/dev/null 2>&1 || {
  echo "未找到 setsid，无法安全启动后台进程组。" >&2
  exit 1
}

nohup setsid bash "${SCRIPT_DIR}/run_vicuna_matrix.sh" >"${MASTER_LOG}" 2>&1 < /dev/null &
matrix_pid=$!
echo "${matrix_pid}" > "${PID_FILE}"

sleep 3
if ! matrix_process_is_running "${matrix_pid}"; then
  echo "Vicuna 生成矩阵启动失败，最近日志如下：" >&2
  tail -n 80 "${MASTER_LOG}" >&2 || true
  exit 1
fi

echo "Vicuna 生成矩阵已在后台启动，PID=${matrix_pid}"
echo "主日志：${MASTER_LOG}"
echo "状态命令：bash ${SCRIPT_DIR}/status_vicuna_matrix.sh"
