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
    echo "Error: --slurm is not yet supported for sdpose installation." >&2
    exit 1
else
    echo "Creating virtual environment at $VENV_DIR ..."
    python3.10 -m venv "$VENV_DIR"
fi


"$VENV_DIR/bin/python" -m pip install --upgrade --no-cache-dir pip setuptools wheel

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$SDPOSE_DIR/requirements.txt"

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$SDPOSE_TOOLS_DIR/pose"
if [ -d "$POSE_REPO" ]; then
    echo "Pose repo already cloned at $POSE_REPO"
else
    git clone -b new_estimators https://github.com/catherine-o-brien/pose.git "$POSE_REPO"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir -e "$POSE_REPO/src/python"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
