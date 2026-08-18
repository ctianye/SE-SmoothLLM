#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/vllm.log}"
BENCHMARK_PID_FILE="${BENCHMARK_PID_FILE:-${PROJECT_DIR}/logs/benchmark.pid}"
BENCHMARK_LOG_FILE="${BENCHMARK_LOG_FILE:-${PROJECT_DIR}/logs/benchmark.log}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/results/raw/jbb-vicuna-13b-gcg-white-box.jsonl}"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== vLLM API =="
curl --fail --silent http://127.0.0.1:8000/v1/models || true
echo
echo "== 主实验进程 =="
if [[ -f "${BENCHMARK_PID_FILE}" ]]; then
  BENCHMARK_PID="$(cat "${BENCHMARK_PID_FILE}")"
  if [[ "${BENCHMARK_PID}" =~ ^[0-9]+$ ]] && kill -0 "${BENCHMARK_PID}" 2>/dev/null; then
    echo "运行中，PID=${BENCHMARK_PID}"
  else
    echo "未运行；PID 文件已过期。"
  fi
else
  echo "尚未启动。"
fi
echo "== 已保存记录 =="
if [[ -f "${OUTPUT}" ]]; then
  wc -l "${OUTPUT}"
else
  echo "0 ${OUTPUT}"
fi
echo "== 主实验最近日志 =="
if [[ -f "${BENCHMARK_LOG_FILE}" ]]; then
  tail -n 20 "${BENCHMARK_LOG_FILE}"
else
  echo "尚无日志。"
fi
echo "== vLLM 最近日志 =="
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 20 "${LOG_FILE}"
else
  echo "尚无日志。"
fi
