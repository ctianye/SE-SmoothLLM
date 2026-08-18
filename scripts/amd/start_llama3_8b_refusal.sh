#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/root/se-smoothllm/repo}"
INPUT_DIR="${INPUT_DIR:-/root/se-smoothllm/results/raw}"
OUTPUT="${OUTPUT:-/root/se-smoothllm/results/judged/jbb-llama3-8b-refusal.jsonl}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
WORKERS="${WORKERS:-8}"
PID_FILE="${STATE_DIR}/llama3-8b-refusal.pid"
LOG_FILE="${LOG_DIR}/llama3-8b-refusal.log"

source "${SCRIPT_DIR}/refusal_judge_common.sh"
mkdir -p "${LOG_DIR}" "${STATE_DIR}" "$(dirname "${OUTPUT}")"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if refusal_judge_process_is_running "${existing_pid}"; then
    echo "Llama-3-8B Refusal Judge 已在运行，PID=${existing_pid}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

command -v setsid >/dev/null 2>&1 || {
  echo "未找到 setsid，无法安全启动后台 Judge。" >&2
  exit 1
}

cd "${PROJECT_DIR}"
nohup setsid python -m benchmarks.run_refusal_judge \
  --input-dir "${INPUT_DIR}" \
  --output "${OUTPUT}" \
  --workers "${WORKERS}" >"${LOG_FILE}" 2>&1 < /dev/null &
judge_pid=$!
echo "${judge_pid}" > "${PID_FILE}"

sleep 3
if ! refusal_judge_process_is_running "${judge_pid}"; then
  echo "Llama-3-8B Refusal Judge 启动失败，最近日志如下：" >&2
  tail -n 80 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Llama-3-8B Refusal Judge 已在后台启动，PID=${judge_pid}"
echo "输出：${OUTPUT}"
echo "日志：${LOG_FILE}"
echo "监控：bash ${SCRIPT_DIR}/watch_llama3_8b_refusal.sh 20"
