#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${ROOT_DIR}/scripts/env_minddrive_b2d.sh"

if [[ -z "${MINDDRIVE_CKPT_PATH:-}" ]]; then
  echo "MINDDRIVE_CKPT_PATH is required for 3B offline latency benchmark." >&2
  echo "Expected a 3B MindDrive task checkpoint, for example a trained rollout/RL checkpoint." >&2
  exit 1
fi

export MINDDRIVE_ENABLE_LATENCY=1
export MINDDRIVE_CAMERA_WIDTH="${MINDDRIVE_CAMERA_WIDTH:-1280}"
export MINDDRIVE_CAMERA_HEIGHT="${MINDDRIVE_CAMERA_HEIGHT:-704}"
export MINDDRIVE_LATENCY_WARMUP_STEPS="${MINDDRIVE_LATENCY_WARMUP_STEPS:-20}"
export MINDDRIVE_KEEP_JPEG_ROUNDTRIP="${MINDDRIVE_KEEP_JPEG_ROUNDTRIP:-1}"
export MINDDRIVE_CONFIG_PATH="${MINDDRIVE_CONFIG_PATH:-${ROOT_DIR}/adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py}"
export MINDDRIVE_LATENCY_SPLIT="${MINDDRIVE_LATENCY_SPLIT:-val}"
export MINDDRIVE_LATENCY_START_INDEX="${MINDDRIVE_LATENCY_START_INDEX:-0}"
export MINDDRIVE_LATENCY_SAMPLE_POOL_SIZE="${MINDDRIVE_LATENCY_SAMPLE_POOL_SIZE:-8}"
export MINDDRIVE_MAX_EGO_FDE="${MINDDRIVE_MAX_EGO_FDE:-20.0}"
export MINDDRIVE_MAX_PATH_FDE="${MINDDRIVE_MAX_PATH_FDE:-25.0}"
export MINDDRIVE_MAX_TRAJ_ABS_M="${MINDDRIVE_MAX_TRAJ_ABS_M:-150.0}"

cmd=(
  "${MINDDRIVE_PYTHON}" "${ROOT_DIR}/scripts/latency/benchmark_minddrive_latency_offline.py"
  --config "${MINDDRIVE_CONFIG_PATH}"
  --checkpoint "${MINDDRIVE_CKPT_PATH}"
  --width "${MINDDRIVE_CAMERA_WIDTH}"
  --height "${MINDDRIVE_CAMERA_HEIGHT}"
  --split "${MINDDRIVE_LATENCY_SPLIT}"
  --start-index "${MINDDRIVE_LATENCY_START_INDEX}"
  --sample-pool-size "${MINDDRIVE_LATENCY_SAMPLE_POOL_SIZE}"
  --max-ego-fde "${MINDDRIVE_MAX_EGO_FDE}"
  --max-path-fde "${MINDDRIVE_MAX_PATH_FDE}"
  --max-traj-abs-m "${MINDDRIVE_MAX_TRAJ_ABS_M}"
  --warmup-steps "${MINDDRIVE_LATENCY_WARMUP_STEPS}"
)

if [[ -n "${MINDDRIVE_OUTPUT_DIR:-}" ]]; then
  cmd+=(--output-dir "${MINDDRIVE_OUTPUT_DIR}")
fi

cmd+=("$@")
"${cmd[@]}"
