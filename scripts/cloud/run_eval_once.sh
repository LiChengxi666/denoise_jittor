#!/usr/bin/env bash
# Run prediction and evaluation for one checkpoint.
# Paths below are interpreted from starter_code/ unless absolute.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STARTER_DIR="${REPO_ROOT}/starter_code"

CHECKPOINT="${CHECKPOINT:-}"
EPOCH="${EPOCH:-}"
PREDICT_NOISY_DIR="${PREDICT_NOISY_DIR:-./val_noisy}"
GT_DIR="${GT_DIR:-./val_gt}"
MESH_DIR="${MESH_DIR:-../dataset_train}"
META_DIR="${META_DIR:-./val_meta}"
PREDICT_DATALIST="${PREDICT_DATALIST:-./datalist/validate.txt}"
PREDICT_CONFIG="${PREDICT_CONFIG:-configs/task/predict_val.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/vm_strong.yaml}"
LOG_ROOT="${LOG_ROOT:-log}"
EXP_NAME="${EXP_NAME:-eval_once}"
GPU_ID="${GPU_ID:-0}"
P2S_NORMALIZE="${P2S_NORMALIZE:-meta}"
EVAL_ENABLED="${EVAL_ENABLED:-1}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/${LOG_ROOT}/${TIMESTAMP}_${EXP_NAME}}"
PRED_DIR="${PRED_DIR:-${LOG_DIR}/predictions/eval_once}"

DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/cloud/run_eval_once.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH
  --epoch N
  --predict-noisy-dir PATH
  --gt-dir PATH
  --mesh-dir PATH
  --meta-dir PATH
  --predict-datalist PATH
  --predict-config PATH
  --model-config PATH
  --pred-dir PATH
  --log-dir PATH
  --gpu-id ID
  --p2s-normalize meta|ref_gt|none
  --no-eval
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --epoch)
      EPOCH="$2"; shift 2 ;;
    --predict-noisy-dir)
      PREDICT_NOISY_DIR="$2"; shift 2 ;;
    --gt-dir)
      GT_DIR="$2"; shift 2 ;;
    --mesh-dir)
      MESH_DIR="$2"; shift 2 ;;
    --meta-dir)
      META_DIR="$2"; shift 2 ;;
    --predict-datalist)
      PREDICT_DATALIST="$2"; shift 2 ;;
    --predict-config)
      PREDICT_CONFIG="$2"; shift 2 ;;
    --model-config)
      MODEL_CONFIG="$2"; shift 2 ;;
    --pred-dir)
      PRED_DIR="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --gpu-id)
      GPU_ID="$2"; shift 2 ;;
    --p2s-normalize)
      P2S_NORMALIZE="$2"; shift 2 ;;
    --no-eval)
      EVAL_ENABLED=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}/config_snapshot/eval_once" "${PRED_DIR}"
EVAL_LOG="${LOG_DIR}/eval.log"
METRICS_CSV="${LOG_DIR}/metrics.csv"
EVAL_JSONL="${LOG_DIR}/eval_results.jsonl"
COMMAND_HISTORY="${LOG_DIR}/command_history.txt"
RUNTIME_DIR="${LOG_DIR}/config_snapshot/eval_once"
PREDICT_DATA_YAML="${RUNTIME_DIR}/predict_data.yaml"
PREDICT_TASK_YAML="${RUNTIME_DIR}/predict_task.yaml"
EVAL_OUTPUT_TXT="${RUNTIME_DIR}/eval_output_${EPOCH:-once}.txt"

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  local cmd=${BASH_COMMAND:-unknown}
  echo "[ERROR] $(date '+%F %T') exit=${exit_code} line=${line_no} cmd=${cmd}" | tee -a "${COMMAND_HISTORY}" >&2
  exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

starter_abs() {
  local path="$1"
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

generate_predict_task() {
  python - "$STARTER_DIR" "$PREDICT_CONFIG" "$MODEL_CONFIG" "$PREDICT_NOISY_DIR" \
    "$PREDICT_DATALIST" "$PRED_DIR" "$PREDICT_TASK_YAML" "$PREDICT_DATA_YAML" <<'PY'
import copy
import os
import sys
import yaml

starter, predict_cfg, model_cfg, noisy_dir, datalist, pred_dir, out_task, out_data = sys.argv[1:9]

def resolve_config(default_dir, name):
    if os.path.isabs(name) or os.path.exists(os.path.join(starter, name)):
        return name if os.path.isabs(name) else os.path.join(starter, name)
    if name.endswith(".yaml"):
        return os.path.join(starter, default_dir, name)
    return os.path.join(starter, default_dir, name + ".yaml")

with open(resolve_config("configs/task", predict_cfg), "r") as f:
    task = yaml.safe_load(f)

data = {
    "predict_dataset": {
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 8,
        "datapath": {
            "input_dataset_dir": noisy_dir,
            "use_prob": False,
            "loader": "npy",
            "data_name": "noisy.npy",
            "ignore_check": True,
            "data_path": {
                "shapenet": [[datalist, 1.0]],
            },
        },
    }
}

task = copy.deepcopy(task)
task["components"] = dict(task["components"])
task["components"]["data"] = out_data
if model_cfg:
    task["components"]["model"] = model_cfg
task["writer"] = dict(task.get("writer") or {})
task["writer"]["__target__"] = task["writer"].get("__target__", "vm")
task["writer"]["save_dir"] = pred_dir
task["writer"]["save_name"] = task["writer"].get("save_name", "denoised")

os.makedirs(os.path.dirname(out_task), exist_ok=True)
with open(out_data, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
with open(out_task, "w") as f:
    yaml.safe_dump(task, f, sort_keys=False, allow_unicode=True)
PY
}

append_metrics() {
  local status="$1"
  local eval_command="$2"
  python - "$METRICS_CSV" "$EVAL_JSONL" "$EVAL_OUTPUT_TXT" "$status" "$EPOCH" \
    "$CHECKPOINT" "$PRED_DIR" "$eval_command" <<'PY'
import csv
import json
import os
import re
import sys
from datetime import datetime

csv_path, jsonl_path, output_path, status, epoch, checkpoint, pred_dir, eval_command = sys.argv[1:9]
text = ""
if os.path.exists(output_path):
    with open(output_path, "r", errors="replace") as f:
        text = f.read()

def parse(patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""

row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "epoch": epoch,
    "checkpoint": checkpoint,
    "pred_dir": pred_dir,
    "status": status,
    "score": parse([r"最终得分.*?([0-9]+(?:\.[0-9]+)?)\s*/\s*100", r"final_score\s*=\s*([0-9]+(?:\.[0-9]+)?)"]),
    "cd_score": parse([r"CD 得分:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100"]),
    "p2s_score": parse([r"P2S 得分:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100"]),
    "mean_cd_pred": parse([r"平均 CD_pred:\s*([0-9.eE+-]+)"]),
    "mean_cd_noisy": parse([r"平均 CD_noisy:\s*([0-9.eE+-]+)"]),
    "mean_p2s_pred": parse([r"平均 P2S_pred:\s*([0-9.eE+-]+)"]),
    "mean_p2s_noisy": parse([r"平均 P2S_noisy:\s*([0-9.eE+-]+)"]),
    "eval_command": eval_command,
}
fieldnames = list(row.keys())
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
with open(jsonl_path, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

if [[ -z "${CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
  echo "--checkpoint is required." >&2
  exit 2
fi

if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(repo_or_starter_abs "${CHECKPOINT}")"
fi

generate_predict_task

echo "Run eval once"
echo "  checkpoint: ${CHECKPOINT:-<dry-run-placeholder>}"
echo "  pred_dir:   ${PRED_DIR}"
echo "  log_dir:    ${LOG_DIR}"

predict_cmd=(
  python run.py
  --task "${PREDICT_TASK_YAML}"
  --load_ckpt "${CHECKPOINT:-DRY_RUN_CHECKPOINT.pkl}"
  --writer_save_dir "${PRED_DIR}"
)

(
  cd "${STARTER_DIR}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  run_logged "${EVAL_LOG}" "${predict_cmd[@]}"
)

noisy_abs="$(starter_abs "${PREDICT_NOISY_DIR}")"
gt_abs="$(starter_abs "${GT_DIR}")"
mesh_abs="$(starter_abs "${MESH_DIR}")"
meta_abs="$(starter_abs "${META_DIR}")"

if [[ "${EVAL_ENABLED}" == "1" && -d "${gt_abs}" ]]; then
  eval_cmd=(
    python evaluate.py
    --pred_dir "${PRED_DIR}"
    --gt_dir "${GT_DIR}"
    --noisy_dir "${PREDICT_NOISY_DIR}"
    --mesh_dir "${MESH_DIR}"
    --meta_dir "${META_DIR}"
    --p2s_normalize "${P2S_NORMALIZE}"
    --csv_path "${LOG_DIR}/per_sample_epoch_${EPOCH:-once}.csv"
    --workers 8
  )
  eval_command_text="$(quote_cmd "${eval_cmd[@]}")"
  (
    cd "${STARTER_DIR}"
    run_logged "${EVAL_LOG}" "${eval_cmd[@]}" | tee "${EVAL_OUTPUT_TXT}" >/dev/null
  )
  append_metrics "ok" "${eval_command_text}"
else
  check_cmd=(
    python tools/check_predictions.py
    --pred_dir "${PRED_DIR}"
    --noisy_dir "${PREDICT_NOISY_DIR}"
  )
  eval_command_text="$(quote_cmd "${check_cmd[@]}")"
  (
    cd "${STARTER_DIR}"
    run_logged "${EVAL_LOG}" "${check_cmd[@]}" | tee "${EVAL_OUTPUT_TXT}" >/dev/null
  )
  append_metrics "checked_no_gt" "${eval_command_text}"
fi

echo "Evaluation record appended:"
echo "  ${METRICS_CSV}"
echo "  ${EVAL_JSONL}"
