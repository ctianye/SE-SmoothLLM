#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PID_FILE="${BENCHMARK_PID_FILE:-${PROJECT_DIR}/logs/benchmark.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "未找到 ${PID_FILE}，无需停止。"
  exit 0
fi

BENCHMARK_PID="$(cat "${PID_FILE}")"
if [[ ! "${BENCHMARK_PID}" =~ ^[0-9]+$ ]]; then
  echo "PID 文件内容无效：${PID_FILE}" >&2
  exit 1
fi
if ! kill -0 "${BENCHMARK_PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "主实验进程已不存在，已清理过期 PID 文件。"
  exit 0
fi

# start_main.sh 使用 setsid，因此负 PID 会向该实验的整个进程组发送信号。
kill -- "-${BENCHMARK_PID}"
for _ in {1..30}; do
  if ! kill -0 "${BENCHMARK_PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "主实验已停止；已完成记录仍保留，可再次启动继续运行。"
    exit 0
  fi
  sleep 1
done

echo "主实验在 30 秒内未退出，请人工检查 PID=${BENCHMARK_PID}。" >&2
exit 1
