#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/42_store/zxz/HUAWEI/VLA"
export CARLA_ROOT="${ROOT_DIR}/carla"
export CUDA_HOME="/usr/local/cuda-11.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${ROOT_DIR}/Bench2Drive:${ROOT_DIR}/Bench2Drive/leaderboard:${ROOT_DIR}/Bench2Drive/leaderboard/team_code:${ROOT_DIR}/Bench2Drive/scenario_runner:${ROOT_DIR}/MindDrive:${PYTHONPATH:-}"

source /home/zxz/anaconda3/etc/profile.d/conda.sh
conda activate /data/zxz/condaenv/minddrive
