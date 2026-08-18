#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL_SECONDS="${1:-20}"
OUTPUT="${OUTPUT:-/root/se-smoothllm/results/judged/jbb-llama3-8b-refusal.jsonl}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
PID_FILE="${STATE_DIR}/llama3-8b-refusal.pid"
LOG_FILE="${LOG_DIR}/llama3-8b-refusal.log"
EXPECTED_RECORDS=1800

source "${SCRIPT_DIR}/refusal_judge_common.sh"

if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "刷新间隔必须是正整数秒。" >&2
  exit 2
fi

trap 'printf "\n已退出监控，后台 Judge 不受影响。\n"; exit 0' INT TERM

while true; do
  clear 2>/dev/null || true
  records=0
  if [[ -f "${OUTPUT}" ]]; then
    records="$(wc -l < "${OUTPUT}")"
  fi
  percent="$(awk -v done="${records}" -v total="${EXPECTED_RECORDS}" \
    'BEGIN { printf "%.1f", done * 100 / total }')"

  process_state="未启动"
  process_elapsed="-"
  if [[ -f "${PID_FILE}" ]]; then
    judge_pid="$(cat "${PID_FILE}")"
    if refusal_judge_process_is_running "${judge_pid}"; then
      process_state="运行中（PID=${judge_pid}）"
      process_elapsed="$(ps -p "${judge_pid}" -o etime= 2>/dev/null | xargs || true)"
    else
      process_state="进程已结束"
    fi
  fi

  echo "JBB Llama-3-8B Refusal Judge 进度"
  echo "更新时间：$(date -Iseconds)"
  echo "进程状态：${process_state}"
  echo "运行时间：${process_elapsed:-未知}"
  echo "总体进度：${records}/${EXPECTED_RECORDS}（${percent}%）"
  if [[ -f "${LOG_FILE}" ]]; then
    echo
    echo "最近日志："
    tail -n 8 "${LOG_FILE}"
  fi
  echo
  echo "按 Ctrl+C 退出监控，不会停止后台 Judge。"
  sleep "${INTERVAL_SECONDS}"
done
