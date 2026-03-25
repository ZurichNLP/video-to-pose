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
    module load cuda/12.6.3 2>/dev/null || echo "CUDA module not found"
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
    BOOTSTRAP_ENV="$MMPOSE_TOOLS_DIR/bootstrap_python3.8"
    source "$(conda info --base)/etc/profile.d/conda.sh"

    echo "Loading miniforge3 module for SLURM..."
    module load miniforge3 2>/dev/null || echo "Warning: miniforge3 module not found"

    # Create bootstrap env if it does not exist
    if [[ ! -x "$BOOTSTRAP_ENV/bin/python" ]]; then
        echo "The cluster's miniforge3 module has only python3.12, and mmposewholebody requires python3.8-3.11."
        echo "Creating temporary Python 3.8 conda environment in which to build the venv..."
        conda create -y --prefix "$BOOTSTRAP_ENV" python=3.8 || exit 1
        echo "Conda environment with python=3.8 successfully created at $BOOTSTRAP_ENV"
    fi

    echo "Activating the conda environment..."
    conda activate $MMPOSE_TOOLS_DIR/bootstrap_python3.8
    echo "Creating venv at $VENV_DIR ..."
    python -m venv $VENV_DIR
    echo "Deactivating conda environment..."
    conda deactivate
    #echo "Removing the temporary conda environment used for bootstrapping..."
    #conda remove --prefix "$BOOTSTRAP_ENV" --all -y
else
    echo "Creating venv at $VENV_DIR ..."
    python3.10 -m venv $VENV_DIR
fi

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
    "$VENV_DIR/bin/pip" install --no-cache-dir \
        "torch==2.1.0" "torchvision==0.16.0"
fi
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$MMPOSEWHOLEBODY_DIR/requirements.txt"

echo "Installing OpenMMLab packages ..."
"$VENV_DIR/bin/pip" install --no-cache-dir openmim
"$VENV_DIR/bin/pip" install --no-cache-dir "mmengine==0.10.7"

# mmcv must come from the OpenMMLab find-links page — PyPI only has a source dist which
# fails to compile. --no-index prevents pip falling back to that source dist.
if [[ "$USE_GPU" == "true" ]]; then
    "$VENV_DIR/bin/pip" install --no-cache-dir --no-index "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
else
    "$VENV_DIR/bin/pip" install --no-cache-dir --no-index "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
fi

"$VENV_DIR/bin/pip" install --no-cache-dir "mmdet==3.3.0"
"$VENV_DIR/bin/pip" install --no-cache-dir "mmpose==1.3.2"

echo
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
