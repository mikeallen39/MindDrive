#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_minddrive_b2d.sh"

"${MINDDRIVE_PYTHON}" -m pip install -U pip
"${MINDDRIVE_PYTHON}" -m pip install \
  black \
  diffusers==0.32.0 \
  stable-baselines3 \
  pyquaternion \
  nuscenes-devkit

"${MINDDRIVE_PYTHON}" "${SCRIPT_DIR}/download_minddrive_latency_assets.py" "$@"
