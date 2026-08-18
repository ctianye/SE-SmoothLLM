#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-autodl}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/benchmarks/configs/vicuna_gcg.json}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/vllm.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm.log}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1200}"

source "${VENV_DIR}/bin/activate"
mkdir -p "${HF_HOME}" "${LOG_DIR}"
export HF_HOME

read_config() {
  python - "${CONFIG_PATH}" "$1" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
value = config
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

MODEL_ID="${MODEL_ID:-$(read_config model.id)}"
MODEL_REVISION="${MODEL_REVISION:-$(read_config model.revision)}"
HOST="${HOST:-$(read_config serving.host)}"
PORT="${PORT:-$(read_config serving.port)}"
DTYPE="${DTYPE:-$(read_config serving.dtype)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(read_config serving.max_model_len)}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$(read_config serving.gpu_memory_utilization)}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(read_config serving.max_num_seqs)}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/vicuna_chat_template.jinja}"

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "vLLM 已在运行，PID=${EXISTING_PID}" >&2
    exit 1
  fi
fi

echo "启动 ${MODEL_ID}@${MODEL_REVISION}，日志：${LOG_FILE}"
nohup vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --generation-config vllm \
  --seed 0 \
  >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!
echo "${VLLM_PID}" > "${PID_FILE}"

DEADLINE=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
until curl --fail --silent "http://${HOST}:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM 启动失败，最后 80 行日志如下：" >&2
    tail -n 80 "${LOG_FILE}" >&2
    exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "等待 vLLM 就绪超时，进程仍在运行；请检查 ${LOG_FILE}。" >&2
    exit 1
  fi
  sleep 5
done

echo "vLLM 已就绪：http://${HOST}:${PORT}/v1"
