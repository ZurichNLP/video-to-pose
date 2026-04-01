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
    BOOTSTRAP_ENV="$SDPOSE_TOOLS_DIR/bootstrap_python3.10"
    source "$(conda info --base)/etc/profile.d/conda.sh"

    echo "Loading miniforge3 module for SLURM..."
    module load miniforge3 2>/dev/null || echo "Warning: miniforge3 module not found"
    module load cuda/12.9.1 2>/dev/null || echo "Warning: cuda module not found"

    # Create bootstrap env if it does not exist
    if [[ ! -x "$BOOTSTRAP_ENV/bin/python" ]]; then
        echo "The cluster's miniforge3 module has only python3.12, and sdpose works best with python3.10"
        echo "Creating temporary Python 3.10 conda environment in which to build the venv..."
        conda create -y --prefix "$BOOTSTRAP_ENV" python=3.10 || exit 1
        echo "Conda environment with python=3.10 successfully created at $BOOTSTRAP_ENV"
    fi

    echo "Activating the conda environment..."
    conda activate $SDPOSE_TOOLS_DIR/bootstrap_python3.10
else
    if command -v python3.10 &>/dev/null; then
        PYTHON_BIN=python3.10
        echo "Creating virtual environment at $VENV_DIR using $PYTHON_BIN..."
    else
        PYTHON_BIN=python3
        echo "python3.10 not available, defaulting to $($PYTHON_BIN --version 2>&1). If you encounter package version errors, retry with python3.10."
    fi

    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

python3.10 -m venv "$VENV_DIR"

if [[ "$USE_SLURM" == "true" ]]; then
    echo "Deactivating conda environment..."
    conda deactivate
    #echo "Removing the temporary conda environment used for bootstrapping..."
    #conda remove --prefix "$BOOTSTRAP_ENV" --all -y
fi


"$VENV_DIR/bin/python" -m pip install --upgrade --no-cache-dir pip setuptools wheel

echo "Installing base requirements ..."
"$VENV_DIR/bin/pip" install --no-cache-dir -r "$SDPOSE_DIR/requirements.txt"

"$VENV_DIR/bin/pip" install "numpy==1.26.4" --force-reinstall

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
