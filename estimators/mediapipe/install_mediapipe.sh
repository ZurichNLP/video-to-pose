#!/usr/bin/env bash
set -euo pipefail

MEDIAPIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MEDIAPIPE_DIR")"
TOOLS=$REPO_DIR/tools

# this script does not change if --slurm is used, but will allow the argument and ignore silently

USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p $TOOLS/mediapipe

VENV_DIR="$TOOLS/mediapipe/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating Python virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$MEDIAPIPE_DIR/requirements.txt"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"