#!/bin/bash

# Pod-side entry point for one training run (models/train.py), launched by
# k8s/kube_launcher.sh through k8s/train_template.yml.
#
# Reads RUN_NAME / MODEL_ID / WORKERS from the env, copies the cached .npy
# dataset off the chips PVC, and trains row "id" == $MODEL_ID of
# models/hparams/$RUN_NAME.json. Checkpoints and logs are copied to
# /model_pvc/$RUN_NAME and /log_pvc/$RUN_NAME, with a .done or .failed marker.

set -euo pipefail

: "${RUN_NAME:?Set RUN_NAME env var, e.g. experiments_1}"
: "${WORKERS:?Set WORKERS env var}"
: "${MODEL_ID:?Set MODEL_ID env var}"

# 10# FORCES BASE 10 -- printf TREATS A LEADING ZERO AS OCTAL, SO A PADDED
# MODEL_ID LIKE 008 ERRORS OUT AND 010 SILENTLY BECOMES 008.
MODEL_ID_PADDED=$(printf "%03d" "$((10#$MODEL_ID))")

# RESOLVED FROM THIS SCRIPT, NOT THE CWD: THE JOB TEMPLATE ALREADY cd's INTO
# THE CLONE BEFORE CALLING US, SO A RELATIVE PATH WOULD LOOK ONE LEVEL TOO DEEP.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHIPS_PVC="/chips_pvc"
MODEL_PVC="/model_pvc"
LOG_PVC="/log_pvc"
CACHE_DIR="/cache"

# train.py --data-dir IS THE DIRECTORY HOLDING {training,validation}_{bands,masks}.npy
DATA_DIR="${CACHE_DIR}"
NET_DIR="${CACHE_DIR}/model"
LOG_DIR="${CACHE_DIR}/logs"
PARAMS_FILE="./hparams/${RUN_NAME}.json"   # RELATIVE TO models/, WHERE train.py RUNS

MODEL_OUT_DIR="${MODEL_PVC}/${RUN_NAME}"
LOG_OUT_DIR="${LOG_PVC}/${RUN_NAME}"

DONE_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.done"
FAIL_MARKER="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}.failed"
STDOUT_LOG_LOCAL="${LOG_DIR}/model_${MODEL_ID_PADDED}_stdout.log"
STDOUT_LOG_PVC="${LOG_OUT_DIR}/model_${MODEL_ID_PADDED}_stdout.log"

mkdir -p "$MODEL_OUT_DIR" "$LOG_OUT_DIR"

# A RERUN STARTS FROM A CLEAN STATE: NO STALE MARKERS, NO OLD STDOUT APPENDED TO
rm -f "$DONE_MARKER" "$FAIL_MARKER" "$STDOUT_LOG_PVC"

echo "Copying dataset files..."
for split in training validation; do
  cp "${CHIPS_PVC}/${split}_bands.npy" "$CACHE_DIR/"
  cp "${CHIPS_PVC}/${split}_masks.npy" "$CACHE_DIR/"
done

mkdir -p "$NET_DIR" "$LOG_DIR"

# train.py IMPORTS models.py AND dataloader.py AS SIBLINGS, SO IT RUNS FROM models/
cd "$REPO_DIR/models"

# MIRROR STDOUT TO THE PVC IN THE BACKGROUND -- tee writes to local disk only,
# so the training loop never blocks on network I/O; this process copies new
# lines over to the PVC as they appear, keeping crash logs visible without
# the per-print latency hit.
touch "$STDOUT_LOG_LOCAL"
tail -F "$STDOUT_LOG_LOCAL" >> "$STDOUT_LOG_PVC" &
TAIL_PID=$!

on_exit() {
  status=$?
  kill "$TAIL_PID" 2>/dev/null || true
  cp "$STDOUT_LOG_LOCAL" "$STDOUT_LOG_PVC" 2>/dev/null || true
  if [ "$status" -eq 0 ]; then
    # FILENAMES CARRY THE EPOCH (best_<id>_e<epoch>.pth.tar), SO cp ALONE WOULD
    # LEAVE A PREVIOUS RUN'S CHECKPOINTS NEXT TO THESE. ONLY DROPPED ON SUCCESS,
    # SO A FAILED RERUN KEEPS THE LAST GOOD ONES.
    rm -f "${MODEL_OUT_DIR}"/*_"${MODEL_ID_PADDED}"_e*.pth.tar
    cp -v "${NET_DIR}"/*.pth.tar "$MODEL_OUT_DIR/" 2>/dev/null || true
    cp -v "${LOG_DIR}"/*.tsv "$LOG_OUT_DIR/" 2>/dev/null || true
    touch "$DONE_MARKER"
  else
    touch "$FAIL_MARKER"
  fi
  exit "$status"
}
trap on_exit EXIT

# kubectl delete SENDS SIGTERM: EXIT THROUGH on_exit SO THE .failed MARKER IS WRITTEN
trap 'exit 143' TERM INT

# BACKGROUND + wait: BASH DEFERS TRAPS UNTIL A FOREGROUND COMMAND RETURNS, BUT
# wait IS INTERRUPTED BY THEM. wait $! STILL RETURNS THE PIPEFAIL STATUS.
python3 -u train.py \
  --data-dir "$DATA_DIR" \
  --net-dir "$NET_DIR" \
  --log-dir "$LOG_DIR" \
  --num-workers "$WORKERS" \
  --params "$PARAMS_FILE" \
  --id "$MODEL_ID" 2>&1 | tee "$STDOUT_LOG_LOCAL" &
wait $!
