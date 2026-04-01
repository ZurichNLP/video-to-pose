#!/usr/bin/env bash
set -euo pipefail

MMPOSEWHOLEBODY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MMPOSEWHOLEBODY_DIR")"
TOOLS=$REPO_DIR/tools

# When --slurm is passed, the script will load the cluster's 
# miniforge3 module to create a temporary conda environment 
# with Python 3.8, which is required to build the venv. 
USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Check if a GPU is available
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected, using CUDA"
    export USE_GPU=true
    # Load CUDA module if needed (cluster-specific)
    module load cuda/12.6.3 2>/dev/null || echo "CUDA module not found. You may need to specify the correct module syntax for your cluster."
else
    echo "No GPU detected, forcing CPU"
    export USE_GPU=false
    export CUDA_VISIBLE_DEVICES=""   # Force CPU
fi

MMPOSE_TOOLS_DIR="$TOOLS/mmposewholebody"
mkdir -p "$MMPOSE_TOOLS_DIR"

VENV_DIR="$MMPOSE_TOOLS_DIR/venv"

if [ -d "$VENV_DIR" ]; then
   echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 0
fi

# Load miniforge3 module if on the cluster (required for venv)
if [[ "$USE_SLURM" == "true" ]]; then
    echo "Loading miniforge3 module for SLURM..."
    module load miniforge3 2>/dev/null || echo "Warning: miniforge3 module not found"
fi

# Prefer python3.12 (newest version supported by torch 2.2.0 / mmpose).
# Fall back to the default python if 3.12 is not available.
if command -v python3.12 &>/dev/null; then
    PYTHON_BIN=python3.12
else
    PYTHON_BIN=python3
    echo "python3.12 not available, defaulting to $($PYTHON_BIN --version 2>&1). If you encounter package version errors, retry with python3.12."
fi
echo "Creating venv at $VENV_DIR using $($PYTHON_BIN --version 2>&1) ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Activating the venv..."
source $VENV_DIR/bin/activate

echo "Cloning pose-format fork (new_estimators branch) ..."
POSE_REPO="$MMPOSE_TOOLS_DIR/pose"
if [ -d "$POSE_REPO" ]; then
    echo "Pose repo already cloned at $POSE_REPO"
else
    git clone -b new_estimators https://github.com/catherine-o-brien/pose.git "$POSE_REPO"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir "$POSE_REPO/src/python"

echo "Installing torch and torchvision ..."
# Versions pinned to match the known-working conda environment (torch 2.1.0+cu118).
if [[ "$USE_GPU" == "true" ]]; then
    "$VENV_DIR/bin/pip" install --no-cache-dir \
        "torch==2.1.0" "torchvision==0.16.0" \
        --index-url https://download.pytorch.org/whl/cu118
else
    # torch 2.2.0 is the minimum that provides Python 3.12 wheels on PyPI
    "$VENV_DIR/bin/pip" install --no-cache-dir \
        "torch==2.2.0" "torchvision==0.17.0"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$MMPOSEWHOLEBODY_DIR/requirements.txt"

echo "Installing OpenMMLab packages ..."
"$VENV_DIR/bin/pip" install --no-cache-dir "mmengine==0.10.7"

# mmcv MUST be installed before openmim.  openmim's dependency chain
# (opendatalab → openxlab) downgrades setuptools to ~60.2.0, which is
# incompatible with Python 3.12 and breaks pkg_resources during any
# subsequent source build.  Installing mmcv first avoids the downgrade.
#
# For GPU: prebuilt wheel from the OpenMMLab find-links page 
# For CPU: pip tries the find-links page first; if no wheel is found it
#   falls back to a PyPI source build, which succeeds because setuptools
#   is still at a Python-3.12-compatible version at this point.
if [[ "$USE_GPU" == "true" ]]; then
    "$VENV_DIR/bin/pip" install --no-cache-dir --no-index "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
else
    "$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir "mmcv==2.1.0"
fi

# chumpy and xtcocotools are mmpose dependencies whose legacy setup.py files
# fail in pip's build isolation environment.  chumpy imports pip directly.
# xtcocotools needs Cython to generate _mask.c from _mask.pyx at build time;
# Cython 3.x breaks the cythonize() call in xtcocotools' setup.py, leaving
# _mask.c missing and the build failing.  Pre-install Cython <3 so the
# generation succeeds, then install both packages without build isolation
# before openmim downgrades setuptools.
"$VENV_DIR/bin/pip" install --no-cache-dir "cython<3"
"$VENV_DIR/bin/pip" install --no-build-isolation --no-cache-dir chumpy xtcocotools

"$VENV_DIR/bin/pip" install --no-cache-dir openmim
"$VENV_DIR/bin/pip" install --no-cache-dir "mmdet==3.3.0"
"$VENV_DIR/bin/pip" install --no-cache-dir "mmpose==1.3.2"

# openmim's dependency chain (opendatalab → openxlab) pins setuptools~=60.2.0,
# whose pkg_resources uses pkgutil.ImpImporter (removed in Python 3.12).
# Some packages also pull in numpy 2.x, which is ABI-incompatible with modules
# compiled against numpy 1.x.  Re-install compatible versions as the final step.
"$VENV_DIR/bin/pip" install --no-cache-dir "setuptools>=67,<71" "numpy<2"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
