#!/usr/bin/env bash
# Pull vm_strong logs/checkpoints from cloud when SSH is configured.
#
# Usage:
#   bash report/scripts/sync_cloud_logs.sh
#   REMOTE=ubuntu@<host> REMOTE_ROOT=/path/to/denoise_jittor bash report/scripts/sync_cloud_logs.sh
#
# After sync, regenerate the report training curve:
#   python report/scripts/plot_training_curve.py
#   cd report && make clean && make

set -euo pipefail

REMOTE="${REMOTE:-ubuntu@36.103.236.211}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/denoise_jittor}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/starter_code"

mkdir -p "${LOCAL_ROOT}/logs/vm_strong" "${LOCAL_ROOT}/experiments/vm_strong"

echo "Sync logs/vm_strong from ${REMOTE}:${REMOTE_ROOT} ..."
rsync -avP "${REMOTE}:${REMOTE_ROOT}/starter_code/logs/vm_strong/" \
  "${LOCAL_ROOT}/logs/vm_strong/"

echo "Sync experiments/vm_strong/checkpoint_*.pkl ..."
rsync -avP "${REMOTE}:${REMOTE_ROOT}/starter_code/experiments/vm_strong/checkpoint_"*.pkl \
  "${LOCAL_ROOT}/experiments/vm_strong/" || true

echo "Sync cloud segmented eval logs (optional) ..."
mkdir -p "${LOCAL_ROOT}/../log"
rsync -avP "${REMOTE}:${REMOTE_ROOT}/log/"*vm_strong* \
  "${LOCAL_ROOT}/../log/" 2>/dev/null || true

echo "Done. Local metrics:"
wc -l "${LOCAL_ROOT}/logs/vm_strong/metrics.csv" || true
ls -lh "${LOCAL_ROOT}/experiments/vm_strong/checkpoint_"*.pkl 2>/dev/null || true
echo
echo "Next: python report/scripts/plot_training_curve.py && cd report && make clean && make"
