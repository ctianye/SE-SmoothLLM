#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
RESULT_DIR="${RESULT_DIR:-/root/se-smoothllm/results/raw}"
PID_FILE="${STATE_DIR}/vicuna-matrix.pid"
MASTER_LOG="${LOG_DIR}/vicuna-matrix.log"

source "${SCRIPT_DIR}/vicuna_matrix_common.sh"

if [[ -f "${PID_FILE}" ]]; then
  matrix_pid="$(cat "${PID_FILE}")"
  if matrix_process_is_running "${matrix_pid}"; then
    echo "状态：运行中，PID=${matrix_pid}"
  else
    echo "状态：进程已结束"
  fi
else
  echo "状态：尚未启动"
fi

if [[ -f "${LOG_DIR}/vicuna-matrix.completed" ]]; then
  echo "完成标记：$(cat "${LOG_DIR}/vicuna-matrix.completed")"
fi

echo "== 已保存记录 =="
shopt -s nullglob
outputs=("${RESULT_DIR}"/*-seed*-*.jsonl)
if [[ "${#outputs[@]}" -eq 0 ]]; then
  echo "尚无结果文件。"
else
  for output in "${outputs[@]}"; do
    printf "%4d %s\n" "$(wc -l < "${output}")" "${output}"
  done
fi

echo "== 最近日志 =="
if [[ -f "${MASTER_LOG}" ]]; then
  tail -n 40 "${MASTER_LOG}"
else
  echo "尚无主日志。"
fi
