#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_DIR="${RESULT_DIR:-/root/se-smoothllm/results/raw}"
LOG_DIR="${LOG_DIR:-/root/se-smoothllm/logs}"
WORKERS="${WORKERS:-4}"
TASK_MAX_ATTEMPTS="${TASK_MAX_ATTEMPTS:-3}"
TASK_RETRY_SECONDS="${TASK_RETRY_SECONDS:-15}"

export HF_HOME="${HF_HOME:-/root/se-smoothllm/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/se-smoothllm/modelscope}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/se-smoothllm/cache}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
rm -f "${LOG_DIR}/vicuna-matrix.completed"
cd "${PROJECT_DIR}"

if [[ ! "${TASK_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASK_MAX_ATTEMPTS 必须是正整数。" >&2
  exit 2
fi
if [[ ! "${TASK_RETRY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "TASK_RETRY_SECONDS 必须是非负整数秒。" >&2
  exit 2
fi

echo "[$(date -Iseconds)] 检查 Vicuna 服务"
python -m benchmarks.probe_backend

splits=(harmful benign)
seeds=(42 43 44)
methods=(undefended smoothllm_fixed se_smoothllm)

for split in "${splits[@]}"; do
  for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
      case "${method}" in
        undefended)
          file_method="undefended"
          ;;
        smoothllm_fixed)
          file_method="smoothllm-fixed"
          ;;
        se_smoothllm)
          file_method="se-smoothllm"
          ;;
        *)
          echo "未知方法：${method}" >&2
          exit 1
          ;;
      esac

      output="${RESULT_DIR}/${split}-seed${seed}-${file_method}.jsonl"
      task_log="${LOG_DIR}/${split}-seed${seed}-${file_method}.log"

      existing_records=0
      if [[ -f "${output}" ]]; then
        existing_records="$(wc -l < "${output}")"
      fi
      if [[ "${existing_records}" -eq 100 ]]; then
        echo "[$(date -Iseconds)] 跳过已完成任务 ${split} seed=${seed} method=${method}"
        continue
      fi
      if [[ "${existing_records}" -gt 100 ]]; then
        echo "任务记录数异常：${output} 预期不超过 100，实际 ${existing_records}" >&2
        exit 1
      fi

      task_succeeded=false
      for ((attempt = 1; attempt <= TASK_MAX_ATTEMPTS; attempt++)); do
        echo "[$(date -Iseconds)] 开始 ${split} seed=${seed} method=${method} attempt=${attempt}/${TASK_MAX_ATTEMPTS}"
        if python -m benchmarks.run_jbb \
          --workers "${WORKERS}" \
          --split "${split}" \
          --seed "${seed}" \
          --method "${method}" \
          --output "${output}" 2>&1 | tee -a "${task_log}"; then
          task_succeeded=true
          break
        fi

        if [[ "${attempt}" -lt "${TASK_MAX_ATTEMPTS}" ]]; then
          echo "[$(date -Iseconds)] 任务失败，${TASK_RETRY_SECONDS} 秒后从检查点重试"
          sleep "${TASK_RETRY_SECONDS}"
        fi
      done

      if [[ "${task_succeeded}" != true ]]; then
        echo "任务连续失败 ${TASK_MAX_ATTEMPTS} 次：${split} seed=${seed} method=${method}" >&2
        exit 1
      fi

      records="$(wc -l < "${output}")"
      if [[ "${records}" -ne 100 ]]; then
        echo "任务记录数异常：${output} 预期 100，实际 ${records}" >&2
        exit 1
      fi
      echo "[$(date -Iseconds)] 完成 ${output}，记录数=${records}"
    done
  done
done

date -Iseconds > "${LOG_DIR}/vicuna-matrix.completed"
echo "全部 Vicuna 生成任务完成。"
