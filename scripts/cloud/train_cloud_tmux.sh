#!/usr/bin/env bash
# Cloud tmux launcher for the point cloud denoising training/eval loop.
# Edit the variables below or override them with environment variables.

set -Eeuo pipefail

SESSION_NAME="${SESSION_NAME:-pc_denoise_train}"
EXP_NAME="${EXP_NAME:-vm_cloud}"
LOG_ROOT="${LOG_ROOT:-log}"
GPU_ID="${GPU_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/${LOG_ROOT}/${TIMESTAMP}_${EXP_NAME}}"
TRAIN_LOOP="${SCRIPT_DIR}/train_eval_loop.sh"

usage() {
  cat <<EOF
Usage:
  bash scripts/cloud/train_cloud_tmux.sh [--dry-run]

Configurable environment variables:
  SESSION_NAME=${SESSION_NAME}
  EXP_NAME=${EXP_NAME}
  LOG_ROOT=${LOG_ROOT}
  GPU_ID=${GPU_ID}

Common overrides:
  TOTAL_EPOCHS=100 EVAL_INTERVAL=10 EXP_NAME=vm_pointfilter \\
    bash scripts/cloud/train_cloud_tmux.sh
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${RUN_DIR}"

COMMAND=(
  bash "${TRAIN_LOOP}"
)

echo "Repository: ${REPO_ROOT}"
echo "Run dir:    ${RUN_DIR}"
echo "Session:    ${SESSION_NAME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] Would launch:"
  printf 'RUN_DIR=%q GPU_ID=%q ' "${RUN_DIR}" "${GPU_ID}"
  printf '%q ' "${COMMAND[@]}"
  echo
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH."
  echo "Fallback command:"
  printf 'RUN_DIR=%q GPU_ID=%q ' "${RUN_DIR}" "${GPU_ID}"
  printf '%q ' "${COMMAND[@]}"
  echo
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  cat <<EOF
tmux session already exists: ${SESSION_NAME}

Attach to the existing session:
  tmux attach -t ${SESSION_NAME}

Or stop it before launching a new run:
  tmux kill-session -t ${SESSION_NAME}

Or use a different name:
  SESSION_NAME=${SESSION_NAME}_2 bash scripts/cloud/train_cloud_tmux.sh
EOF
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -c "${REPO_ROOT}" \
  "RUN_DIR=$(printf '%q' "${RUN_DIR}") GPU_ID=$(printf '%q' "${GPU_ID}") bash $(printf '%q' "${TRAIN_LOOP}")"

cat <<EOF
Started tmux session: ${SESSION_NAME}

Attach with:
  tmux attach -t ${SESSION_NAME}

Detach inside tmux with:
  Ctrl-b then d

Logs:
  ${RUN_DIR}/train.log
  ${RUN_DIR}/eval.log
EOF
