#!/usr/bin/env bash
set -euo pipefail

SAPIENS_DIR="$(cd "$(dirname "$0")" && pwd)"
ESTIMATORS_DIR="$(dirname "$SAPIENS_DIR")"
REPO_DIR="$(dirname "$ESTIMATORS_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$USE_SLURM" == "true" ]]; then
    echo "Loading miniforge3 module for SLURM..."
    module load miniforge3 2>/dev/null || echo "Warning: miniforge3 module not found"
fi

# Check if a GPU is available
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected, using CUDA"
    if [[ "$USE_SLURM" == "true" ]]; then
        module load cuda/12.6.3 2>/dev/null || echo "CUDA module not found. You may need to specify the correct module syntax for your cluster."
    fi
    USE_GPU=true
else
    echo "No GPU detected, forcing CPU"
    USE_GPU=false
fi

SAPIENS_TOOLS_DIR="$TOOLS/sapiens"
mkdir -p "$SAPIENS_TOOLS_DIR"

VENV_DIR="$SAPIENS_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating virtual environment for Sapiens at $VENV_DIR with $(python --version) ..."
python -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$SAPIENS_DIR/requirements.txt"

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$SAPIENS_TOOLS_DIR/pose"
if [ -d "$POSE_REPO" ]; then
    echo "Pose repo already cloned at $POSE_REPO"
else
    git clone -b new_estimators https://github.com/catherine-o-brien/pose.git "$POSE_REPO"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir -e "$POSE_REPO/src/python[sapiens]"

# Pre-download the Sapiens pose model.
# Upstream renamed the file on HF main (AP_640 → AP_639), so sapiens_inference's
# download path (raw requests.get on a constructed main-branch URL) 404s on a
# fresh runner.  Pin a snapshot that still hosts AP_640, and save as ".pt"
# (no "2") — that's what download_hf_model() checks for as the "already cached"
# sentinel before trying to download.
SAPIENS_PKG_DIR="$("$VENV_DIR/bin/python" -c \
    'import os, sapiens_inference; print(os.path.dirname(os.path.abspath(sapiens_inference.__file__)))')"
SAPIENS_REPO_ROOT="$(dirname "$SAPIENS_PKG_DIR")"
SAPIENS_MODELS_DIR="$SAPIENS_REPO_ROOT/models"
mkdir -p "$SAPIENS_MODELS_DIR"
SAPIENS_MODEL_DST="$SAPIENS_MODELS_DIR/sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt"
SAPIENS_MODEL_URL="https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/4caa2b2290255dc8963b5ead35fe3c6e761742aa/sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"
if [ ! -f "$SAPIENS_MODEL_DST" ]; then
    echo "Pre-downloading Sapiens pose model to $SAPIENS_MODEL_DST ..."
    wget -q --show-progress -O "$SAPIENS_MODEL_DST" "$SAPIENS_MODEL_URL"
fi
