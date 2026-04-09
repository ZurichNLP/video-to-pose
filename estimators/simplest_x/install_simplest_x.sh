#!/usr/bin/env bash
set -euo pipefail

SIMPLEST_X_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SIMPLEST_X_DIR")"
TOOLS=$REPO_DIR/tools

# --slurm is accepted and silently ignored (no container build needed)
USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p $TOOLS/simplest_x

SMPLEST_X_REPO=$TOOLS/simplest_x/SMPLest-X

if [[ ! -d "$SMPLEST_X_REPO" ]]; then
    git clone --branch pose_estimation_study \
        https://github.com/GerrySant/SMPLest-X.git "$SMPLEST_X_REPO"
fi

VENV_DIR="$TOOLS/simplest_x/venv"

if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating Python virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing PyTorch with CUDA 12.1 support ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip
"$VENV_DIR/bin/pip" install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "Installing Python dependencies ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$SIMPLEST_X_DIR/requirements.txt"

echo "Installing pose-format (multiple_support branch) ..."
"$VENV_DIR/bin/pip" install --no-cache-dir \
    "git+https://github.com/GerrySant/pose.git@multiple_support#subdirectory=src/python"

echo "Installing pytest ..."
"$VENV_DIR/bin/pip" install --no-cache-dir pytest

echo "Downloading pretrained models ..."
"$VENV_DIR/bin/python" "$SIMPLEST_X_DIR/download_models.py" "$SMPLEST_X_REPO"

echo
echo "=== Setup complete ==="
echo "Repo: $SMPLEST_X_REPO"
echo "Venv: $VENV_DIR"
