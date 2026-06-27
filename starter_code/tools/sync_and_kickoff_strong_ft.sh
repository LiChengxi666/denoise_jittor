#!/usr/bin/env bash
# Run from LOCAL WSL after configs are ready.
# Syncs code to cloud and prints the tmux train command.

set -euo pipefail

REMOTE="ubuntu@36.103.236.211"
REMOTE_ROOT="/home/ubuntu/denoise_jittor"

rsync -avP \
  --exclude='.git' \
  --exclude='dataset_train' \
  --exclude='dataset_test_noisy' \
  --exclude='starter_code/experiments' \
  --exclude='starter_code/tmp_predict' \
  --exclude='starter_code/result.zip' \
  "/home/enovoczy/计图/" \
  "${REMOTE}:${REMOTE_ROOT}/"

echo
echo "=== Cloud: verify checkpoint_39 and start training ==="
echo "ssh ${REMOTE}"
echo "cd ${REMOTE_ROOT}/starter_code"
echo "source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate jittor"
echo "ls -lh experiments/vm_strong/checkpoint_39.pkl"
echo "tmux new -s strong_ft"
echo "bash tools/run_strong_ft_from39.sh train"
echo
echo "=== After 30 epochs: grid + predict ==="
echo "bash tools/run_strong_ft_from39.sh grid"
echo "# edit configs/task/predict_vm_strong_indclean1600_ft.yaml load_ckpt to grid best"
echo "bash tools/run_strong_ft_from39.sh predict"
