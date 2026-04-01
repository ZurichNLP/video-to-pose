#!/usr/bin/env bash
set -euo pipefail

OPENPIFPAF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$OPENPIFPAF_DIR")"
TOOLS=$REPO_DIR/tools

# --slurm is accepted but has no effect for this estimator
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
    module load cuda/12.6.3 2>/dev/null || echo "CUDA module not found. You may need to specify the correct module syntax for your cluster."
    USE_GPU=true
else
    echo "No GPU detected, forcing CPU"
    USE_GPU=false
fi

OPENPIFPAF_TOOLS_DIR="$TOOLS/openpifpaf"
mkdir -p "$OPENPIFPAF_TOOLS_DIR"

VENV_DIR="$OPENPIFPAF_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

PYTHON=""
# openpifpaf requires torch 1.13.1 / torchvision 0.14.1, which only have wheels for Python <=3.10
for candidate in python3.10 python3.9; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Error: openpifpaf requires Python 3.9 or 3.10, but neither was found on PATH." >&2
    exit 1
fi

echo "Creating Python virtual environment at $VENV_DIR using $PYTHON ..."
"$PYTHON" -m venv "$VENV_DIR"

echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$OPENPIFPAF_DIR/requirements.txt"

echo "Installing torch ..."
# torch must be installed before openpifpaf: openpifpaf's pyproject.toml pins
# torch==1.13.1 as a build dependency, which is no longer on PyPI. Pre-installing
# torch and using --no-build-isolation skips that constraint.
if [[ "$USE_GPU" == "true" ]]; then
    "$VENV_DIR/bin/pip" install --no-cache-dir 'torch==1.13.1' 'torchvision==0.14.1' \
        --index-url https://download.pytorch.org/whl/cu118
else
    "$VENV_DIR/bin/pip" install --no-cache-dir 'torch==1.13.1' 'torchvision==0.14.1'
fi

echo "Installing openpifpaf ..."
# setuptools is needed for pkg_resources (used by torch 1.13.1 cpp_extension during build)
"$VENV_DIR/bin/pip" install --no-cache-dir 'setuptools<70' wheel ninja
"$VENV_DIR/bin/pip" install --no-cache-dir --no-build-isolation openpifpaf

echo "Installing opencv-python-headless ..."
# install together with numpy<2 constraint so pip picks a compatible opencv version
"$VENV_DIR/bin/pip" install --no-cache-dir opencv-python-headless 'numpy<2'

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$OPENPIFPAF_TOOLS_DIR/pose"
if [ -d "$POSE_REPO" ]; then
    echo "Pose repo already cloned at $POSE_REPO"
else
    git clone -b new_estimators https://github.com/catherine-o-brien/pose.git "$POSE_REPO"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir -e "$POSE_REPO/src/python"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo
echo "Note: the shufflenetv2k30-wholebody checkpoint (~100 MB) will be downloaded"
echo "automatically on first run."
