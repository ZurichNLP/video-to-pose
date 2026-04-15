#!/usr/bin/env bash
set -euo pipefail

SDPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SDPOSE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SDPOSE_TOOLS_DIR="$TOOLS/sdpose"
mkdir -p "$SDPOSE_TOOLS_DIR"

VENV_DIR="$SDPOSE_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

if [ "$USE_SLURM" = true ]; then
    module load miniforge3 2>/dev/null || echo "Warning: miniforge3 module not found"
fi

# Check if a GPU is available
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected, using CUDA"
    if [[ "$USE_SLURM" == "true" ]]; then
        module load cuda/12.9.1 2>/dev/null || echo "CUDA module not found. You may need to specify the correct module syntax for your cluster."
    fi
else
    echo "No GPU detected, forcing CPU"
fi

if command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
    echo "Creating virtual environment at $VENV_DIR using $PYTHON_BIN..."
else
    PYTHON_BIN=python3
    echo "python3.12 not available, defaulting to $($PYTHON_BIN --version 2>&1). If you encounter package version errors, retry with python3.12."
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install "cython<3" "setuptools>=67,<71" "numpy<2" 
"$VENV_DIR/bin/pip" install --no-build-isolation chumpy xtcocotools
"$VENV_DIR/bin/pip" install -r "$SDPOSE_DIR/requirements.txt"

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$SDPOSE_TOOLS_DIR/pose"
if [ -d "$POSE_REPO" ]; then
    echo "Pose repo already cloned at $POSE_REPO"
else
    git clone -b new_estimators https://github.com/catherine-o-brien/pose.git "$POSE_REPO"
fi
"$VENV_DIR/bin/pip" install "$POSE_REPO/src/python"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
