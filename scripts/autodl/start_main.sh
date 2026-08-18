#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
PID_FILE="${BENCHMARK_PID_FILE:-${LOG_DIR}/benchmark.pid}"
LOG_FILE="${BENCHMARK_LOG_FILE:-${LOG_DIR}/benchmark.log}"

mkdir -p "${LOG_DIR}"
if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "主实验已在运行，PID=${EXISTING_PID}" >&2
    exit 1
  fi
fi

command -v setsid >/dev/null 2>&1 || {
  echo "未找到 setsid，无法创建可独立停止的后台进程组。" >&2
  exit 1
}
nohup setsid bash "${SCRIPT_DIR}/run_main.sh" >"${LOG_FILE}" 2>&1 &
BENCHMARK_PID=$!
echo "${BENCHMARK_PID}" > "${PID_FILE}"
sleep 2
if ! kill -0 "${BENCHMARK_PID}" 2>/dev/null; then
  echo "主实验未能启动，日志如下：" >&2
  tail -n 80 "${LOG_FILE}" >&2
  exit 1
fi

echo "主实验已在后台运行，PID=${BENCHMARK_PID}，日志=${LOG_FILE}"
