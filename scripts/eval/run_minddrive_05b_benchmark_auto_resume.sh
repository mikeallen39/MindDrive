#!/usr/bin/env bash

set -euo pipefail

EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EVAL_SCRIPT_DIR}/../.." && pwd)"

MODEL_GPU="${1:-2}"
ROUTES_BASENAME="${2:-${BENCH2DRIVE_ROOT:-${ROOT_DIR}/Bench2Drive}/leaderboard/data/bench2drive220}"
PORT="${3:-30000}"
TM_PORT="${4:-50000}"
CARLA_ADAPTER="${5:-3}"

source "${ROOT_DIR}/scripts/env_minddrive_b2d.sh"

CHECKPOINT_DIR="${MINDDRIVE_CHECKPOINT_DIR:-minddrive_05b_results}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_DIR}/$(basename "${ROUTES_BASENAME}").json"
SUPERVISOR_LOG="${MINDDRIVE_SUPERVISOR_LOG:-}"
MAX_RESTARTS="${MINDDRIVE_MAX_RESTARTS:-50}"
RESTART_DELAY="${MINDDRIVE_RESTART_DELAY:-15}"
MAX_STAGNANT_ATTEMPTS="${MINDDRIVE_MAX_STAGNANT_ATTEMPTS:-5}"
STRICT_PORTS="${MINDDRIVE_STRICT_PORTS:-1}"
EVALUATOR_PATTERN="--checkpoint=${CHECKPOINT_ENDPOINT}"

if [[ "${CHECKPOINT_ENDPOINT}" = /* ]]; then
  CHECKPOINT_ABS="${CHECKPOINT_ENDPOINT}"
else
  CHECKPOINT_ABS="${ROOT_DIR}/${CHECKPOINT_ENDPOINT}"
fi

CMD=(
  env "MINDDRIVE_STRICT_PORTS=${STRICT_PORTS}"
  "${MINDDRIVE_PYTHON}" "${EVAL_SCRIPT_DIR}/run_benchmark_supervisor.py"
  --runner "${EVAL_SCRIPT_DIR}/run_minddrive_05b_benchmark.sh"
  --checkpoint "${CHECKPOINT_ABS}"
  --max-restarts "${MAX_RESTARTS}"
  --restart-delay "${RESTART_DELAY}"
  --max-stagnant-attempts "${MAX_STAGNANT_ATTEMPTS}"
  "--kill-pattern=-carla-rpc-port=${PORT}"
  "--kill-pattern=${EVALUATOR_PATTERN}"
)

if [[ -n "${SUPERVISOR_LOG}" ]]; then
  CMD+=(--log-file "${SUPERVISOR_LOG}")
fi

CMD+=(-- "${MODEL_GPU}" "${ROUTES_BASENAME}" "${PORT}" "${TM_PORT}" "${CARLA_ADAPTER}")

"${CMD[@]}"
