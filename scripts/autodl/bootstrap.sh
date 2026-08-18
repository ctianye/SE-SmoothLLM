#!/usr/bin/env bash
set -euo pipefail

# 只负责安装和自检，不启动模型，便于先确认环境再开始长任务。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-autodl}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "未找到 nvidia-smi，请确认实例已分配 NVIDIA GPU。" >&2
  exit 1
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "未找到 ${PYTHON_BIN}。" >&2
  exit 1
}

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== 磁盘 =="
df -h "${PROJECT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"需要 Python >= 3.10，当前为 {sys.version.split()[0]}")
print(f"Python {sys.version.split()[0]}")
PY

mkdir -p "${HF_HOME}" "${PROJECT_DIR}/results/runtime"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_DIR}[dev,benchmark]"
python -m pip install -r "${PROJECT_DIR}/requirements-autodl.txt"

export HF_HOME
python -c "import se_smoothllm, vllm; print('se_smoothllm 和 vllm 导入成功')"
python -m pytest -q
python -m pip check
python -m pip freeze > "${PROJECT_DIR}/results/runtime/pip-freeze.txt"

echo "环境准备完成。HF_HOME=${HF_HOME}"
