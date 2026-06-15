#!/usr/bin/env bash
# Run the two baseline CPU latency configs sequentially, then print a summary.
# Sequential on purpose: single-thread spec must not be polluted by other
# python procs stealing core time.

set -euo pipefail

cd "$(dirname "$0")/.."  # cd to SPACE_PROJECT root

PY="${PY:-python}"
SCRIPT="sgc_cpu_latency/latency_cpu.py"
WEIGHTS_512="ML_CHOICE/v2_R11_MNv4-S_512.pth"
WEIGHTS_1024="ML_CHOICE/v2_R11_MNv4-S_1024.pth"

echo
echo "########################################################"
echo "# CPU latency — baselines only (R11 @ 512 and @ 1024)"
echo "########################################################"

# 1) v2_R11_MNv4-S_512 baseline (deployable model)
$PY $SCRIPT \
    --mode baseline \
    --weights "$WEIGHTS_512" \
    --input-hw 512 \
    --run-name v2_R11_MNv4-S_512_baseline

# 2) v2_R11_MNv4-S_1024 baseline (research ceiling — comparison)
$PY $SCRIPT \
    --mode baseline \
    --weights "$WEIGHTS_1024" \
    --input-hw 1024 \
    --run-name v2_R11_MNv4-S_1024_baseline

echo
echo "########################################################"
echo "# Summary"
echo "########################################################"
$PY sgc_cpu_latency/summarize.py
