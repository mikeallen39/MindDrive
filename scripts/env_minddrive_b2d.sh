#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MINDDRIVE_ROOT="${ROOT_DIR}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/rl_projects:${ROOT_DIR}/rl_projects/scenario_runner:${ROOT_DIR}/team_code:${PYTHONPATH:-}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
    "/usr/local/cuda-11.8" \
    "/usr/local/cuda"
  do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${CARLA_ROOT:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/carla" \
    "${ROOT_DIR}/../carla" \
    "/cache/carla" \
    "/data/carla" \
    "/home/ma-user/work/carla" \
    "/home/carla"
  do
    if [[ -d "${candidate}" ]]; then
      export CARLA_ROOT="${candidate}"
      break
    fi
  done
fi

export CARLA_ROOT="${CARLA_ROOT:-${ROOT_DIR}/carla}"

if [[ -z "${BENCH2DRIVE_ROOT:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/Bench2Drive" \
    "${ROOT_DIR}/../Bench2Drive"
  do
    if [[ -d "${candidate}" ]]; then
      export BENCH2DRIVE_ROOT="${candidate}"
      break
    fi
  done
fi

ASCEND_SET_ENV_SH="${ASCEND_SET_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_SET_ENV_SH}" ]]; then
  # torch_npu relies on these environment variables to expose the NPU backend.
  # shellcheck disable=SC1090
  source "${ASCEND_SET_ENV_SH}"
fi

if [[ -d "${CARLA_ROOT}/PythonAPI" ]]; then
  export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"
  if compgen -G "${CARLA_ROOT}/PythonAPI/carla/dist/carla-*.egg" > /dev/null; then
    for egg_path in "${CARLA_ROOT}"/PythonAPI/carla/dist/carla-*.egg; do
      export PYTHONPATH="${egg_path}:${PYTHONPATH}"
    done
  fi
fi

if [[ -z "${MINDDRIVE_PYTHON:-}" ]]; then
  if [[ -x "/data/zxz/condaenv/minddrive/bin/python" ]]; then
    export MINDDRIVE_PYTHON="/data/zxz/condaenv/minddrive/bin/python"
  elif [[ -x "/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/python" ]]; then
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
