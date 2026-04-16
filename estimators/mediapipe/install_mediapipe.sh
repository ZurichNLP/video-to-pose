#!/usr/bin/env bash
set -euo pipefail

MEDIAPIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
ESTIMATORS_DIR="$(dirname "$MEDIAPIPE_DIR")"
REPO_DIR="$(dirname "$ESTIMATORS_DIR")"
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
# mediapipe<0.10.30 has no macOS wheels for Python 3.12+, so on macOS we need 3.11 or 3.10.
# On Linux, Python 3.12 has wheels available, so we use whatever python3 is.
PYTHON="python3"
if [[ "$(uname -s)" == "Darwin" ]]; then
    PYTHON=""
    for candidate in python3.11 python3.10; do
        if command -v "$candidate" &>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
    if [[ -z "$PYTHON" ]]; then
        echo "Error: Python 3.11 or 3.10 is required on macOS (mediapipe<0.10.30 has no macOS wheels for Python 3.12+)." >&2
        echo "Install with: brew install python@3.11" >&2
        exit 1
    fi
fi
echo "Using $PYTHON"
"$PYTHON" -m venv "$VENV_DIR"

echo "Installing dependencies ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$MEDIAPIPE_DIR/requirements.txt"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"