#!/usr/bin/env bash
set -euo pipefail

MMPOSEWHOLEBODY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MMPOSEWHOLEBODY_DIR")"
TOOLS=$REPO_DIR/tools

# --slurm is accepted but has no effect for this estimator
USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

MMPOSE_TOOLS_DIR="$TOOLS/mmposewholebody"
mkdir -p "$MMPOSE_TOOLS_DIR"

VENV_DIR="$MMPOSE_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

echo "Creating Python virtual environment at $VENV_DIR ..."
# mmcv wheels are only published for Python 3.8-3.11; Python 3.12+ has no
# pre-built wheels and building from source fails due to pkg_resources removal.
# Require Python 3.10 or 3.11.
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "Error: Python 3.11 or 3.10 is required (mmcv has no wheels for Python 3.12+)." >&2
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "Install with: brew install python@3.11" >&2
    else
        echo "Install with: sudo apt install python3.11   # Debian/Ubuntu" >&2
        echo "          or: sudo dnf install python3.11   # RHEL/Fedora" >&2
        echo "          or: module load python/3.11       # HPC module system" >&2
    fi
    exit 1
fi
echo "Using $PYTHON"
"$PYTHON" -m venv "$VENV_DIR"

echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --no-cache-dir --upgrade pip

echo "Installing PyTorch 2.1 (CPU build; pinned to match available mmcv wheels) ..."
# Torch is pinned to 2.1.0 because the OpenMMLab wheel index (used below for
# mmcv) publishes wheels per torch minor version, and 2.1 is the newest that
# has confirmed macOS ARM CPU wheels.
"$VENV_DIR/bin/pip" install --no-cache-dir "torch==2.1.0" "torchvision==0.16.0"

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$MMPOSEWHOLEBODY_DIR/requirements.txt"

echo "Installing mmpose stack ..."
# chumpy (a mmpose dependency) has an ancient setup.py that does 'import pip',
# which pip's isolated build environment doesn't provide. Install it first with
# --no-build-isolation so it can see the venv's pip, then mmpose finds it already
# satisfied and skips the broken build.
"$VENV_DIR/bin/pip" install --no-cache-dir --no-build-isolation chumpy
"$VENV_DIR/bin/pip" install --no-cache-dir mmengine "mmdet>=3.1.0" mmpose
# mmcv does not publish pre-built wheels on PyPI for all platforms; source builds
# fail because pkg_resources is absent from pip's isolated build env with newer
# setuptools. Use the OpenMMLab wheel index instead (CPU build works on both
# macOS and Linux). For GPU inference on Linux, reinstall mmcv after install:
#   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
#   (replace cu118 with your CUDA version: cu117, cu121, etc.)
"$VENV_DIR/bin/pip" install --no-cache-dir "mmcv==2.1.0" \
    -f "https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html"
# mmpose's dependency resolution upgrades numpy to 2.x, but xtcocotools was
# compiled against numpy 1.x and is binary-incompatible with numpy 2.x.
# Re-pin numpy<2 after the full mmpose stack is installed.
echo "Re-pinning numpy<2 (mmpose upgrades numpy to 2.x, breaking xtcocotools) ..."
"$VENV_DIR/bin/pip" install --no-cache-dir "numpy<2"

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$MMPOSE_TOOLS_DIR/pose"
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
echo "Note: PyTorch 2.1.0 (CPU) was installed."
echo "For GPU inference, reinstall torch and mmcv for your CUDA version:"
echo "  https://pytorch.org/get-started/locally/"
echo "  https://mmcv.readthedocs.io/en/latest/get_started/installation.html"
