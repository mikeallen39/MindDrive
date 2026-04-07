#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${ROOT_DIR}"

MODEL_GPU="${1:-2}"
ROUTES_BASENAME="${2:-leaderboard/data/bench2drive220}"
PORT="${3:-30000}"
TM_PORT="${4:-50000}"
CARLA_ADAPTER="${5:-3}"

source "${SCRIPT_DIR}/env_minddrive_b2d.sh"

cd "${BENCH_DIR}"

export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}/rl_projects:${ROOT_DIR}/team_code"
export SCENARIO_RUNNER_ROOT="${ROOT_DIR}/rl_projects/scenario_runner"
export LEADERBOARD_ROOT="${ROOT_DIR}/rl_projects/leaderboard"
export CHALLENGE_TRACK_CODENAME="SENSORS"

TEAM_AGENT="${ROOT_DIR}/team_code/minddrive_b2d_agent.py"
CONFIG_PATH="${MINDDRIVE_CONFIG_PATH:-${ROOT_DIR}/adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py}"
CKPT_PATH="${MINDDRIVE_CKPT_PATH:-${ROOT_DIR}/ckpts/minddrive_rltrain.pth}"
TEAM_CONFIG="${CONFIG_PATH}+${CKPT_PATH}"
CHECKPOINT_DIR="${MINDDRIVE_CHECKPOINT_DIR:-minddrive_05b_results}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_DIR}/$(basename "${ROUTES_BASENAME}").json"
SAVE_DIR="${MINDDRIVE_SAVE_DIR:-eval_minddrive_05b_traj}"
SAVE_PATH="${BENCH_DIR}/${SAVE_DIR}"
PLANNER_TYPE="${MINDDRIVE_PLANNER_TYPE:-traj}"
IS_BENCH2DRIVE="True"
ROUTES_XML="${ROUTES_BASENAME}.xml"

export PORT="${PORT}"
export TM_PORT="${TM_PORT}"
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE="${IS_BENCH2DRIVE}"
export PLANNER_TYPE="${PLANNER_TYPE}"
export GPU_RANK="${CARLA_ADAPTER}"
export ROUTES="${ROUTES_XML}"
export TEAM_AGENT="${TEAM_AGENT}"
export TEAM_CONFIG="${TEAM_CONFIG}"
export CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT}"
export SAVE_PATH="${SAVE_PATH}"

mkdir -p "$(dirname "${CHECKPOINT_ENDPOINT}")" "${SAVE_PATH}"

"${MINDDRIVE_PYTHON}" "${SCRIPT_DIR}/check_minddrive_carla_env.py" --require-runtime

"${MINDDRIVE_PYTHON}" "${LEADERBOARD_ROOT}/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --repetitions="${REPETITIONS}" \
  --track="${CHALLENGE_TRACK_CODENAME}" \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug="${DEBUG_CHALLENGE}" \
  --record="${RECORD_PATH:-}" \
  --resume="${RESUME}" \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}"
