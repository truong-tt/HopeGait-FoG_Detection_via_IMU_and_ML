#!/usr/bin/env bash
# One-shot cloud-GPU launcher.
#
# Assumes:
#   - You're SSHed into a CUDA box (RunPod / Lambda / Vast / your own GPU).
#   - The repo is checked out at the current directory.
#   - Raw data is at data/raw/  OR  preprocessed data is at data/processed/.
#
# Usage:
#   bash scripts/launch_cloud.sh                # full pipeline, all windows
#   WINDOW=200 bash scripts/launch_cloud.sh     # one window, faster
#   SUBJECT=03 bash scripts/launch_cloud.sh     # one subject, smoke
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARN: no NVIDIA driver found. Falling back to CPU (very slow)."
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
if command -v nvidia-smi >/dev/null 2>&1; then
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [ -d "data/raw" ] && [ ! -d "data/processed" ]; then
    echo "Preprocessing raw data..."
    python src/data_pipeline/preprocess.py
fi

ARGS=""
if [ -n "${WINDOW:-}" ]; then ARGS="$ARGS --window $WINDOW"; fi
if [ -n "${SUBJECT:-}" ]; then ARGS="$ARGS --subject $SUBJECT"; fi
if [ -n "${EPOCHS:-}" ]; then export HOPEGAIT_EPOCHS="$EPOCHS"; fi

echo "Training with args: $ARGS"
python src/training/train.py $ARGS
python src/training/evaluate.py

echo "Done. Models in models/, summary in models/evaluation_summary.json."
