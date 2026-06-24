#!/usr/bin/env bash
# Segmented cloud training loop with periodic prediction and evaluation.
# Paths below are interpreted from starter_code/ unless documented otherwise.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STARTER_DIR="${REPO_ROOT}/starter_code"

# ---- User-editable defaults. Override any of these with environment variables. ----
TRAIN_DATA="${TRAIN_DATA:-../dataset_train}"
VALIDATE_DATA="${VALIDATE_DATA:-${TRAIN_DATA}}"
PREDICT_NOISY_DIR="${PREDICT_NOISY_DIR:-./val_noisy}"
GT_DIR="${GT_DIR:-./val_gt}"
MESH_DIR="${MESH_DIR:-../dataset_train}"
META_DIR="${META_DIR:-./val_meta}"
PREDICT_DATALIST="${PREDICT_DATALIST:-./datalist/validate.txt}"
TRAIN_DATALIST="${TRAIN_DATALIST:-}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/task/train_vm.yaml}"
PREDICT_CONFIG="${PREDICT_CONFIG:-configs/task/predict_val.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/vm_strong.yaml}"

LOG_ROOT="${LOG_ROOT:-log}"
EXP_NAME="${EXP_NAME:-vm_cloud}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
GPU_ID="${GPU_ID:-0}"
P2S_NORMALIZE="${P2S_NORMALIZE:-meta}"
RESUME_CKPT="${RESUME_CKPT:-}"
EVAL_ENABLED="${EVAL_ENABLED:-1}"
TRAIN_VALIDATE_EVERY="${TRAIN_VALIDATE_EVERY:-1}"
SEED="${SEED:-123}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/${LOG_ROOT}/${TIMESTAMP}_${EXP_NAME}}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/cloud/train_eval_loop.sh [--dry-run]

Important environment overrides:
  EXP_NAME=${EXP_NAME}
  TOTAL_EPOCHS=${TOTAL_EPOCHS}
  EVAL_INTERVAL=${EVAL_INTERVAL}
  GPU_ID=${GPU_ID}
  TRAIN_DATA=${TRAIN_DATA}
  PREDICT_NOISY_DIR=${PREDICT_NOISY_DIR}
  GT_DIR=${GT_DIR}
  RESUME_CKPT=${RESUME_CKPT:-<empty>}
EOF
}

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

mkdir -p "${RUN_DIR}"/{config_snapshot,checkpoints,predictions}
TRAIN_LOG="${RUN_DIR}/train.log"
EVAL_LOG="${RUN_DIR}/eval.log"
METRICS_CSV="${RUN_DIR}/metrics.csv"
EVAL_JSONL="${RUN_DIR}/eval_results.jsonl"
COMMAND_HISTORY="${RUN_DIR}/command_history.txt"

exec > >(tee -a "${TRAIN_LOG}") 2>&1

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  local cmd=${BASH_COMMAND:-unknown}
  {
    echo "[ERROR] $(date '+%F %T') exit=${exit_code} line=${line_no} cmd=${cmd}"
  } | tee -a "${COMMAND_HISTORY}" >&2
  exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

starter_abs() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${STARTER_DIR}/${path}"
  fi
}

repo_or_starter_abs() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  elif [[ -e "${REPO_ROOT}/${path}" ]]; then
    printf '%s\n' "${REPO_ROOT}/${path}"
  else
    printf '%s\n' "${STARTER_DIR}/${path}"
  fi
}

quote_cmd() {
  printf '%q ' "$@"
}

record_cmd() {
  echo "[$(date '+%F %T')] $(quote_cmd "$@")" | tee -a "${COMMAND_HISTORY}" >/dev/null
}

run_logged() {
  local log_file="$1"
  shift
  record_cmd "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $(quote_cmd "$@")" | tee -a "${log_file}"
    return 0
  fi
  "$@" 2>&1 | tee -a "${log_file}"
  return "${PIPESTATUS[0]}"
}

write_environment_snapshot() {
  {
    echo "# Cloud training environment"
    echo "start_time=$(date '+%F %T')"
    echo "repo_root=${REPO_ROOT}"
    echo "starter_dir=${STARTER_DIR}"
    echo "run_dir=${RUN_DIR}"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "git_status_short_begin"
    git -C "${REPO_ROOT}" status --short 2>/dev/null || true
    echo "git_status_short_end"
    echo "python=$(command -v python || true)"
    python --version 2>&1 || true
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
    echo "nvidia_smi_begin"
    nvidia-smi 2>/dev/null || echo "nvidia-smi unavailable"
    echo "nvidia_smi_end"
  } | tee "${RUN_DIR}/environment.txt"
}

generate_train_task() {
  local segment_epochs="$1"
  local segment_tag="$2"
  local ckpt_dir="$3"
  local out_task="$4"
  local out_data="$5"
  local load_ckpt="$6"

  python - "$STARTER_DIR" "$TRAIN_CONFIG" "$MODEL_CONFIG" "$TRAIN_DATA" "$VALIDATE_DATA" \
    "$TRAIN_DATALIST" "$BATCH_SIZE" "$NUM_WORKERS" "$segment_epochs" "$EVAL_INTERVAL" \
    "$TRAIN_VALIDATE_EVERY" "$out_task" "$out_data" "$load_ckpt" <<'PY'
import copy
import os
import sys
import yaml

starter, train_cfg, model_cfg, train_data, validate_data, train_datalist = sys.argv[1:7]
batch_size, num_workers, segment_epochs, eval_interval, validate_every = sys.argv[7:12]
out_task, out_data, load_ckpt = sys.argv[12:15]

def resolve_config(default_dir, name):
    if os.path.isabs(name) or os.path.exists(os.path.join(starter, name)):
        return name if os.path.isabs(name) else os.path.join(starter, name)
    if name.endswith(".yaml"):
        return os.path.join(starter, default_dir, name)
    return os.path.join(starter, default_dir, name + ".yaml")

with open(resolve_config("configs/task", train_cfg), "r") as f:
    task = yaml.safe_load(f)

data_name = task["components"]["data"]
with open(resolve_config("configs/data", data_name), "r") as f:
    data = yaml.safe_load(f)

def update_dataset(ds, root):
    if not ds:
        return
    ds.setdefault("datapath", {})["input_dataset_dir"] = root
    if batch_size:
        ds["batch_size"] = int(batch_size)
    if num_workers:
        ds["num_workers"] = int(num_workers)
    if train_datalist and "data_path" in ds.get("datapath", {}):
        keys = list(ds["datapath"]["data_path"].keys())
        if keys:
            ds["datapath"]["data_path"][keys[0]] = [[train_datalist, 1.0]]

update_dataset(data.get("train_dataset"), train_data)
update_dataset(data.get("validate_dataset"), validate_data)

task = copy.deepcopy(task)
task["components"] = dict(task["components"])
task["components"]["data"] = out_data
if model_cfg:
    task["components"]["model"] = model_cfg
if load_ckpt:
    task["load_ckpt"] = load_ckpt
else:
    task.pop("load_ckpt", None)

trainer = dict(task.get("trainer") or {})
trainer["epochs"] = int(segment_epochs)
trainer["save_every"] = int(segment_epochs)
trainer["save_last"] = True
trainer["save_last_every"] = 1
trainer["validate_every"] = int(validate_every)
trainer["log_to_file"] = True
trainer["log_config"] = True
task["trainer"] = trainer

os.makedirs(os.path.dirname(out_task), exist_ok=True)
with open(out_data, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
with open(out_task, "w") as f:
    yaml.safe_dump(task, f, sort_keys=False, allow_unicode=True)
PY
}

write_environment_snapshot

if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_CKPT="$(repo_or_starter_abs "${RESUME_CKPT}")"
fi

echo "Run directory: ${RUN_DIR}"
echo "Total epochs: ${TOTAL_EPOCHS}"
echo "Eval interval: ${EVAL_INTERVAL}"
echo "Weight-level resume checkpoint: ${RESUME_CKPT:-<none>}"

if [[ "${TOTAL_EPOCHS}" -le 0 || "${EVAL_INTERVAL}" -le 0 ]]; then
  echo "TOTAL_EPOCHS and EVAL_INTERVAL must be positive integers." >&2
  exit 2
fi

current_epoch=0
current_ckpt="${RESUME_CKPT}"

while [[ "${current_epoch}" -lt "${TOTAL_EPOCHS}" ]]; do
  next_epoch=$(( current_epoch + EVAL_INTERVAL ))
  if [[ "${next_epoch}" -gt "${TOTAL_EPOCHS}" ]]; then
    next_epoch="${TOTAL_EPOCHS}"
  fi
  segment_epochs=$(( next_epoch - current_epoch ))
  segment_tag="$(printf 'epoch_%03d' "${next_epoch}")"
  segment_dir="${RUN_DIR}/config_snapshot/${segment_tag}"
  segment_ckpt_dir="${RUN_DIR}/checkpoints/${segment_tag}"
  segment_task="${segment_dir}/train_task.yaml"
  segment_data="${segment_dir}/train_data.yaml"
  mkdir -p "${segment_dir}" "${segment_ckpt_dir}"

  echo
  echo "========== Segment ${segment_tag}: train ${segment_epochs} epoch(s), target cumulative epoch ${next_epoch} =========="
  generate_train_task "${segment_epochs}" "${segment_tag}" "${segment_ckpt_dir}" \
    "${segment_task}" "${segment_data}" "${current_ckpt}"

  train_cmd=(
    python run.py
    --task "${segment_task}"
    --seed "${SEED}"
    --ckpt_dir "${segment_ckpt_dir}"
    --log_dir "${RUN_DIR}/internal_train_logs"
    --experiment_name "${segment_tag}"
  )
  (
    cd "${STARTER_DIR}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    run_logged "${TRAIN_LOG}" "${train_cmd[@]}"
  )

  current_ckpt="${segment_ckpt_dir}/checkpoint_last.pkl"
  if [[ "${DRY_RUN}" != "1" && ! -f "${current_ckpt}" ]]; then
    echo "Expected checkpoint not found: ${current_ckpt}" >&2
    exit 1
  fi

  echo "Segment checkpoint: ${current_ckpt}"
  eval_cmd=(
    bash "${SCRIPT_DIR}/run_eval_once.sh"
    --checkpoint "${current_ckpt}"
    --epoch "${next_epoch}"
    --log-dir "${RUN_DIR}"
    --pred-dir "${RUN_DIR}/predictions/${segment_tag}"
    --predict-noisy-dir "${PREDICT_NOISY_DIR}"
    --gt-dir "${GT_DIR}"
    --mesh-dir "${MESH_DIR}"
    --meta-dir "${META_DIR}"
    --predict-datalist "${PREDICT_DATALIST}"
    --predict-config "${PREDICT_CONFIG}"
    --model-config "${MODEL_CONFIG}"
    --gpu-id "${GPU_ID}"
    --p2s-normalize "${P2S_NORMALIZE}"
  )
  if [[ "${EVAL_ENABLED}" != "1" ]]; then
    eval_cmd+=(--no-eval)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    eval_cmd+=(--dry-run)
  fi
  run_logged "${EVAL_LOG}" "${eval_cmd[@]}"

  current_epoch="${next_epoch}"
done

echo
echo "Training/evaluation loop finished at $(date '+%F %T')"
echo "Run directory: ${RUN_DIR}"
