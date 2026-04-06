#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MINDDRIVE_ROOT="${ROOT_DIR}"
export CARLA_ROOT="${CARLA_ROOT:-${ROOT_DIR}/carla}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/rl_projects:${ROOT_DIR}/rl_projects/scenario_runner:${ROOT_DIR}/team_code:${PYTHONPATH:-}"

ASCEND_SET_ENV_SH="${ASCEND_SET_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_SET_ENV_SH}" ]]; then
  # torch_npu relies on these environment variables to expose the NPU backend.
  # shellcheck disable=SC1090
  source "${ASCEND_SET_ENV_SH}"
fi

if [[ -d "${CARLA_ROOT}/PythonAPI" ]]; then
  export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"
fi

if [[ -z "${MINDDRIVE_PYTHON:-}" ]]; then
  if [[ -x "/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/python" ]]; then
    export MINDDRIVE_PYTHON="/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/python"
  elif [[ -x "/home/ma-user/anaconda3/envs/minddrive-npu-latency/bin/python" ]]; then
    export MINDDRIVE_PYTHON="/home/ma-user/anaconda3/envs/minddrive-npu-latency/bin/python"
  elif command -v python >/dev/null 2>&1; then
    export MINDDRIVE_PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    export MINDDRIVE_PYTHON="$(command -v python3)"
  else
    echo "python interpreter not found" >&2
    exit 1
  fi
fi
