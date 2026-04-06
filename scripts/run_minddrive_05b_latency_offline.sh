#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_minddrive_b2d.sh"

export MINDDRIVE_ENABLE_LATENCY=1
export MINDDRIVE_CAMERA_WIDTH="${MINDDRIVE_CAMERA_WIDTH:-1280}"
export MINDDRIVE_CAMERA_HEIGHT="${MINDDRIVE_CAMERA_HEIGHT:-704}"
export MINDDRIVE_LATENCY_WARMUP_STEPS="${MINDDRIVE_LATENCY_WARMUP_STEPS:-20}"
export MINDDRIVE_KEEP_JPEG_ROUNDTRIP="${MINDDRIVE_KEEP_JPEG_ROUNDTRIP:-1}"
export MINDDRIVE_CONFIG_PATH="${MINDDRIVE_CONFIG_PATH:-${ROOT_DIR}/adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py}"
export MINDDRIVE_CKPT_PATH="${MINDDRIVE_CKPT_PATH:-${ROOT_DIR}/ckpts/minddrive_rltrain.pth}"
export MINDDRIVE_OUTPUT_DIR="${MINDDRIVE_OUTPUT_DIR:-${ROOT_DIR}/results_latency_offline_1280x704}"

"${MINDDRIVE_PYTHON}" "${SCRIPT_DIR}/benchmark_minddrive_latency_offline.py" \
  --config "${MINDDRIVE_CONFIG_PATH}" \
  --checkpoint "${MINDDRIVE_CKPT_PATH}" \
  --width "${MINDDRIVE_CAMERA_WIDTH}" \
  --height "${MINDDRIVE_CAMERA_HEIGHT}" \
  --warmup-steps "${MINDDRIVE_LATENCY_WARMUP_STEPS}" \
  --output-dir "${MINDDRIVE_OUTPUT_DIR}" \
  "$@"
