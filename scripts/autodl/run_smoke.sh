#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-autodl}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${SMOKE_OUTPUT:-${PROJECT_DIR}/results/raw/smoke-${TIMESTAMP}.jsonl}"

cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"
python -m benchmarks.probe_backend
python -m benchmarks.run_jbb \
  --method undefended \
  --method smoothllm_fixed \
  --method se_smoothllm \
  --seed 42 \
  --split harmful \
  --split benign \
  --limit 1 \
  --workers 2 \
  --output "${OUTPUT}"

echo "Smoke test 完成：${OUTPUT}"
