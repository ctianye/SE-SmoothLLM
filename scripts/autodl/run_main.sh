#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-autodl}"
WORKERS="${WORKERS:-8}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/results/raw/jbb-vicuna-13b-gcg-white-box.jsonl}"

cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"
python -m benchmarks.run_jbb --workers "${WORKERS}" --output "${OUTPUT}"
