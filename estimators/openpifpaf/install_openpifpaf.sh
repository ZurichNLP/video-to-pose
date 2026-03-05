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

OPENPIFPAF_TOOLS_DIR="$TOOLS/openpifpaf"
mkdir -p "$OPENPIFPAF_TOOLS_DIR"

VENV_DIR="$OPENPIFPAF_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating Python virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$OPENPIFPAF_DIR/requirements.txt"

echo "Installing torch ..."
# torch must be installed before openpifpaf: openpifpaf's pyproject.toml pins
# torch==1.13.1 as a build dependency, which is no longer on PyPI. Pre-installing
# torch and using --no-build-isolation skips that constraint.
"$VENV_DIR/bin/pip" install --no-cache-dir torch torchvision

echo "Installing openpifpaf ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --no-build-isolation openpifpaf

echo "Installing opencv-python-headless ..."
"$VENV_DIR/bin/pip" install --no-cache-dir opencv-python-headless

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
