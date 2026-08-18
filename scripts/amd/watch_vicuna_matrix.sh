#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL_SECONDS="${1:-20}"
STATE_DIR="${STATE_DIR:-/root/se-smoothllm/run-state}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
RESULT_DIR="${RESULT_DIR:-/root/se-smoothllm/results/raw}"
PID_FILE="${STATE_DIR}/vicuna-matrix.pid"
MASTER_LOG="${LOG_DIR}/vicuna-matrix.log"
EXPECTED_FILES=18
EXPECTED_RECORDS=1800

source "${SCRIPT_DIR}/vicuna_matrix_common.sh"

if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "刷新间隔必须是正整数秒。" >&2
  exit 2
fi

file_records() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

method_file_name() {
  case "$1" in
    undefended)
      echo "undefended"
      ;;
    smoothllm_fixed)
      echo "smoothllm-fixed"
      ;;
    se_smoothllm)
      echo "se-smoothllm"
      ;;
  esac
}

trap 'printf "\n已退出监控，后台实验不受影响。\n"; exit 0' INT TERM

while true; do
  clear 2>/dev/null || true
  echo "SE-SmoothLLM Vicuna 实验进度"
  echo "更新时间：$(date -Iseconds)"

  process_state="未启动"
  process_elapsed="-"
  if [[ -f "${PID_FILE}" ]]; then
    matrix_pid="$(cat "${PID_FILE}")"
    if matrix_process_is_running "${matrix_pid}"; then
      process_state="运行中（PID=${matrix_pid}）"
      process_elapsed="$(ps -p "${matrix_pid}" -o etime= 2>/dev/null | xargs || true)"
    else
      process_state="进程已结束"
    fi
  fi
  echo "进程状态：${process_state}"
  echo "本次后台运行时间：${process_elapsed:-未知}"

  total_records=0
  completed_files=0
  splits=(harmful benign)
  seeds=(42 43 44)
  methods=(undefended smoothllm_fixed se_smoothllm)

  echo
  echo "各组合记录数（目标均为 100）"
  for split in "${splits[@]}"; do
    for seed in "${seeds[@]}"; do
      counts=()
      for method in "${methods[@]}"; do
        file_method="$(method_file_name "${method}")"
        output="${RESULT_DIR}/${split}-seed${seed}-${file_method}.jsonl"
        records="$(file_records "${output}")"
        total_records=$((total_records + records))
        if [[ "${records}" -eq 100 ]]; then
          completed_files=$((completed_files + 1))
        fi
        counts+=("${records}")
      done
      printf "%-7s seed=%s  undefended=%3d  fixed=%3d  SE=%3d\n" \
        "${split}" "${seed}" "${counts[0]}" "${counts[1]}" "${counts[2]}"
    done
  done

  percent="$(awk -v done="${total_records}" -v total="${EXPECTED_RECORDS}" \
    'BEGIN { printf "%.1f", done * 100 / total }')"
  bar_width=40
  filled=$((total_records * bar_width / EXPECTED_RECORDS))
  empty=$((bar_width - filled))
  filled_bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  empty_bar="$(printf '%*s' "${empty}" '' | tr ' ' '-')"

  echo
  printf "总体进度：[%s%s] %d/%d（%s%%）\n" \
    "${filled_bar}" "${empty_bar}" "${total_records}" "${EXPECTED_RECORDS}" "${percent}"
  echo "完成文件：${completed_files}/${EXPECTED_FILES}"

  if [[ -f "${LOG_DIR}/vicuna-matrix.completed" ]]; then
    echo "完成标记：$(cat "${LOG_DIR}/vicuna-matrix.completed")"
  elif [[ -f "${MASTER_LOG}" ]]; then
    current_task="$(grep -E '开始 (harmful|benign)' "${MASTER_LOG}" | tail -n 1 || true)"
    if [[ -n "${current_task}" ]]; then
      echo "当前任务：${current_task}"
    fi
  fi

  echo
  echo "按 Ctrl+C 退出监控，不会停止后台实验。"
  sleep "${INTERVAL_SECONDS}"
done
