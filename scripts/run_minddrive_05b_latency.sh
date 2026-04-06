#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_GPU="${1:-2}"
ROUTES_BASENAME="${2:-leaderboard/data/drivetransformer_bench2drive_dev10}"
PORT="${3:-30000}"
TM_PORT="${4:-50000}"
CARLA_ADAPTER="${5:-3}"

export MINDDRIVE_ENABLE_LATENCY=1
export MINDDRIVE_CAMERA_WIDTH=1280
export MINDDRIVE_CAMERA_HEIGHT=704
export MINDDRIVE_LATENCY_WARMUP_STEPS="${MINDDRIVE_LATENCY_WARMUP_STEPS:-20}"
export MINDDRIVE_KEEP_JPEG_ROUNDTRIP="${MINDDRIVE_KEEP_JPEG_ROUNDTRIP:-1}"
export MINDDRIVE_CONFIG_PATH="${ROOT_DIR}/adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py"
export MINDDRIVE_CHECKPOINT_DIR="${MINDDRIVE_CHECKPOINT_DIR:-minddrive_05b_latency_results}"
export MINDDRIVE_SAVE_DIR="${MINDDRIVE_SAVE_DIR:-eval_minddrive_05b_latency_1280x704}"

"${SCRIPT_DIR}/run_minddrive_05b_benchmark.sh" \
  "${MODEL_GPU}" \
  "${ROUTES_BASENAME}" \
  "${PORT}" \
  "${TM_PORT}" \
  "${CARLA_ADAPTER}"
