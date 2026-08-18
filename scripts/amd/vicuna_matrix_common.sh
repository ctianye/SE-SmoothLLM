#!/usr/bin/env bash

matrix_process_is_running() {
  local matrix_pid="$1"
  local process_state
  local process_command

  [[ "${matrix_pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${matrix_pid}" 2>/dev/null || return 1

  process_state="$(ps -p "${matrix_pid}" -o stat= 2>/dev/null | xargs || true)"
  [[ -n "${process_state}" && "${process_state}" != Z* ]] || return 1

  process_command="$(ps -p "${matrix_pid}" -o args= 2>/dev/null || true)"
  [[ "${process_command}" == *"run_vicuna_matrix.sh"* ]]
}
