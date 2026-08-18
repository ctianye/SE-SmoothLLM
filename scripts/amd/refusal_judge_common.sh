#!/usr/bin/env bash

refusal_judge_process_is_running() {
  local judge_pid="$1"
  local process_state
  local process_command

  [[ "${judge_pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${judge_pid}" 2>/dev/null || return 1
  process_state="$(ps -p "${judge_pid}" -o stat= 2>/dev/null | xargs || true)"
  [[ -n "${process_state}" && "${process_state}" != Z* ]] || return 1
  process_command="$(ps -p "${judge_pid}" -o args= 2>/dev/null || true)"
  [[ "${process_command}" == *"benchmarks.run_refusal_judge"* ]]
}
