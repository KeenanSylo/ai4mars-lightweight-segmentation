#!/bin/bash
# Chained pipeline for ITERATION-2 / 003_fpn_512:
#   1. train  → 2. evaluate (Strategy A)  → 3. space_grade (--input-hw 512)
#
# Designed to be launched once with nohup so it survives VSCode / shell close.
# Each stage's stdout+stderr goes to its own .log file inside the experiment dir.
# A PIPELINE_DONE marker is written at the end iff every stage succeeded;
# PIPELINE_FAILED.<stage> is written if any stage exits non-zero so the failure
# point is obvious when Claude returns.

set -u
cd "$(dirname "$0")/../../.."   # → .
ROOT="$(pwd)"
EXP_ID="003_fpn_512"
EXP_DIR="ITERATION-2/experiments/${EXP_ID}"
VENV="${ROOT}/.venv/bin/python"
STATUS_DIR="${EXP_DIR}"

date_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

mkdir -p "${EXP_DIR}"
PIPELINE_LOG="${EXP_DIR}/pipeline.log"

{
  echo "==========================================================================="
  echo "Pipeline start: $(date_iso)"
  echo "Working dir:    ${ROOT}"
  echo "Experiment id:  ${EXP_ID}"
  echo "Python:         ${VENV}"
  echo "==========================================================================="
} > "${PIPELINE_LOG}"

# ----- Stage 1: training ----------------------------------------------------
{
  echo
  echo "[STAGE 1/3] Training FPN + tu-mobilenetv3_small_100 @ 512x512"
  echo "  start: $(date_iso)"
} >> "${PIPELINE_LOG}"

"${VENV}" FPN_training_v2.py \
    --exp-id "${EXP_ID}" \
    --input-hw 512 \
    --description "FPN + MNv3-Small-100 at 512x512 (RQ1 cross-architecture vs 002_dlv3plus_512). Same training recipe; only decoder family differs (FPN vs DeepLabV3+)." \
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
