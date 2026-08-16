#!/usr/bin/env bash
# Fine-tune from vm_strong checkpoint_39 (cd0.3 + indclean1600, 30 epochs).
# Run on cloud: cd ~/denoise_jittor/starter_code && bash tools/run_strong_ft_from39.sh [train|grid|predict|all]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_NAME="vm_strong_cd03_indclean1600_ft"
CKPT_DIR="experiments/${EXP_NAME}"
BASELINE_CKPT="experiments/vm_strong/checkpoint_39.pkl"
TASK="configs/task/train_vm_strong_cd03_indclean1600_ft.yaml"
PREDICT_TASK="configs/task/predict_vm_strong_indclean1600_ft.yaml"
GRID_CSV="diagnostics/${EXP_NAME}_grid.csv"
COMPARE_CSV="diagnostics/${EXP_NAME}_vs_baseline39.csv"

setup_train_env() {
  unset LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export MALLOC_ARENA_MAX=2
}

run_train() {
  setup_train_env
  if [[ ! -f "$BASELINE_CKPT" ]]; then
    echo "ERROR: missing $BASELINE_CKPT"
    exit 1
  fi
  echo "Starting fine-tune from $BASELINE_CKPT -> $CKPT_DIR"
  python run.py \
    --task "$TASK" \
    --ckpt_dir "$CKPT_DIR" \
    --experiment_name "$EXP_NAME"
}

run_grid() {
  mkdir -p diagnostics
  python tools/eval_predict_grid.py \
    --checkpoints "${CKPT_DIR}/checkpoint_*.pkl" \
    --base_task configs/task/predict_val.yaml \
    --base_model configs/model/vm_strong_indclean1600.yaml \
    --step_sizes 0.8 \
    --predict_steps 2 \
    --inner_steps 4 \
    --patch_sizes 1200 \
    --alpha_blends 1.0 \
    --momentums 0.0 \
    --step_decays none \
    --csv_path "$GRID_CSV"

  python tools/eval_predict_grid.py \
    --checkpoints "$BASELINE_CKPT" \
    --base_task configs/task/predict_val.yaml \
    --base_model configs/model/vm_strong.yaml \
    --step_sizes 0.8 \
    --predict_steps 2 \
    --inner_steps 4 \
    --patch_sizes 1200 \
    --alpha_blends 1.0 \
    --momentums 0.0 \
    --step_decays none \
    --csv_path "diagnostics/baseline39_verify.csv"

  echo "Grid results: $GRID_CSV"
  echo "Baseline verify: diagnostics/baseline39_verify.csv"
  python tools/apply_grid_best_ckpt.py \
    --csv "$GRID_CSV" \
    --predict_task "$PREDICT_TASK" \
    --p2s_floor 80.37
}

run_predict() {
  if [[ ! -f "$PREDICT_TASK" ]]; then
    echo "ERROR: missing $PREDICT_TASK"
    exit 1
  fi
  python run.py --task "$PREDICT_TASK"
  python tools/check_predictions.py \
    --pred_dir ./tmp_predict \
    --noisy_dir ../dataset_test_noisy
  (cd tmp_predict && zip -r ../result.zip shapenet/)
  ls -lh ../result.zip
}

case "${1:-train}" in
  train) run_train ;;
  grid) run_grid ;;
  predict) run_predict ;;
  all)
    run_train
    run_grid
    run_predict
    ;;
  *)
    echo "Usage: $0 [train|grid|predict|all]"
    exit 1
    ;;
esac
