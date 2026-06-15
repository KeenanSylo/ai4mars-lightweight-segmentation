#!/bin/bash
# Chained pipeline for ITERATION-2 / 006_dlv3plus_mnv4_conv_small_512:
#   1. train  → 2. evaluate (Strategy A)  → 3. space_grade (--input-hw 512)
#
# Runs on GPU 0 via CUDA_VISIBLE_DEVICES=0 in parallel with 005 (on GPU 1).
# Same nohup pattern as 003/004/005.
#
# Encoder choice: tu-mobilenetv4_conv_small (newest mainstream CNN successor to
# MobileNetV3-Small; 2024 release by Google). Total params with DLV3+ decoder
# = 2,999,220 = 100.0% of the 3 M cap (the most cap-saturating pure-CNN option).
# Tests "newer-CNN modernity" effect vs 002's MobileNetV3-Small at the same
# decoder, encoder family, resolution, and training recipe.

set -u
cd "$(dirname "$0")/../../.."   # → .
ROOT="$(pwd)"
EXP_ID="006_dlv3plus_mnv4_conv_small_512"
EXP_DIR="ITERATION-2/experiments/${EXP_ID}"
VENV="${ROOT}/.venv/bin/python"
STATUS_DIR="${EXP_DIR}"

# Bind to GPU 0 — keep entire pipeline (training + space_grade GPU latency)
# on the same device. GPU 1 is in use by 005; GPU 3 is in use by another user.
export CUDA_VISIBLE_DEVICES=0

date_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

mkdir -p "${EXP_DIR}"
PIPELINE_LOG="${EXP_DIR}/pipeline.log"

{
  echo "==========================================================================="
  echo "Pipeline start: $(date_iso)"
  echo "Working dir:    ${ROOT}"
  echo "Experiment id:  ${EXP_ID}"
  echo "Python:         ${VENV}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "==========================================================================="
} > "${PIPELINE_LOG}"

# ----- Stage 1: training ----------------------------------------------------
{
  echo
  echo "[STAGE 1/3] Training DeepLabV3+ + tu-mobilenetv4_conv_small @ 512x512 (GPU 0)"
  echo "  start: $(date_iso)"
} >> "${PIPELINE_LOG}"

"${VENV}" DeepLabV3Plus_training_v2.py \
    --exp-id "${EXP_ID}" \
    --encoder tu-mobilenetv4_conv_small \
    --input-hw 512 \
    --description "DeepLabV3+ + MobileNetV4-conv-small at 512x512 (newer-CNN modernity test vs 002's MobileNetV3-Small). 3.00 M params = 100.0% of 3 M cap. Tests whether the 2024 mobile-CNN succeeds the 2019 mobile-CNN on AI4Mars at the same decoder, resolution, and training recipe." \
    > "${EXP_DIR}/training.log" 2>&1
RC=$?
if [ ${RC} -ne 0 ]; then
    echo "[STAGE 1/3] FAILED (exit ${RC}) at $(date_iso)" >> "${PIPELINE_LOG}"
    echo "training failed (rc=${RC})" > "${STATUS_DIR}/PIPELINE_FAILED.train"
    exit ${RC}
fi
echo "[STAGE 1/3] OK at $(date_iso)" >> "${PIPELINE_LOG}"

# ----- Stage 2: evaluate (Strategy A — auto-reads input_hw from config.json) -
{
  echo
  echo "[STAGE 2/3] Evaluating on gold test set (Strategy A upsample)"
  echo "  start: $(date_iso)"
} >> "${PIPELINE_LOG}"

"${VENV}" evaluate.py --exp-id "${EXP_ID}" \
    > "${EXP_DIR}/evaluate.log" 2>&1
RC=$?
if [ ${RC} -ne 0 ]; then
    echo "[STAGE 2/3] FAILED (exit ${RC}) at $(date_iso)" >> "${PIPELINE_LOG}"
    echo "evaluate failed (rc=${RC})" > "${STATUS_DIR}/PIPELINE_FAILED.evaluate"
    exit ${RC}
fi
echo "[STAGE 2/3] OK at $(date_iso)" >> "${PIPELINE_LOG}"

# ----- Stage 3: space_grade at --input-hw 512 -------------------------------
{
  echo
  echo "[STAGE 3/3] Space-grade scorecard (--input-hw 512)"
  echo "  start: $(date_iso)"
} >> "${PIPELINE_LOG}"

"${VENV}" space_grade.py --exp-id "${EXP_ID}" --input-hw 512 \
    > "${EXP_DIR}/space_grade.log" 2>&1
RC=$?
if [ ${RC} -ne 0 ]; then
    echo "[STAGE 3/3] FAILED (exit ${RC}) at $(date_iso)" >> "${PIPELINE_LOG}"
    echo "space_grade failed (rc=${RC})" > "${STATUS_DIR}/PIPELINE_FAILED.space_grade"
    exit ${RC}
fi
echo "[STAGE 3/3] OK at $(date_iso)" >> "${PIPELINE_LOG}"

# ----- Success marker -------------------------------------------------------
{
  echo
  echo "==========================================================================="
  echo "Pipeline complete: $(date_iso)"
  echo "All three stages succeeded. Read training.log / evaluate.log / space_grade.log"
  echo "for stage output; space_grade.json + evaluation_results.json hold structured"
  echo "results."
  echo "==========================================================================="
} >> "${PIPELINE_LOG}"

echo "$(date_iso)" > "${STATUS_DIR}/PIPELINE_DONE"
