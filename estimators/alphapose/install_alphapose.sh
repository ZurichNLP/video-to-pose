#!/usr/bin/env bash
set -euo pipefail

ALPHAPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$ALPHAPOSE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p $TOOLS/alphapose

ALPHAPOSE_SINGULARITY_DIR=$TOOLS/alphapose/alphapose-singularity-uzh

if [[ ! -d "$ALPHAPOSE_SINGULARITY_DIR" ]]; then
    git clone https://github.com/bricksdont/alphapose-singularity-uzh $ALPHAPOSE_SINGULARITY_DIR
fi

# Pull the Singularity container image (~8GB)
# Login nodes may not have enough memory; use --slurm to submit as a SLURM job instead.
if [ "$USE_SLURM" = true ]; then
    # slurm_build_container.sh uses $SLURM_SUBMIT_DIR to locate scripts/, so sbatch
    # must be called from the repo root.
    (cd $ALPHAPOSE_SINGULARITY_DIR && sbatch scripts/slurm_build_container.sh)
    echo "Container build submitted as SLURM job. Monitor with: squeue -u \$USER"
else
    if [[ ! -f "$ALPHAPOSE_SINGULARITY_DIR/alphapose.sif" ]]; then
        if command -v apptainer &>/dev/null; then
            SINGULARITY_CMD="apptainer"
        elif command -v singularity &>/dev/null; then
            SINGULARITY_CMD="singularity"
        else
            echo "Error: neither apptainer nor singularity found. Please install one of them first." >&2
            exit 1
        fi
        $SINGULARITY_CMD pull $ALPHAPOSE_SINGULARITY_DIR/alphapose.sif \
            oras://ghcr.io/bricksdont/alphapose-singularity-uzh/alphapose:latest
    fi
fi

# Set up Python virtual environment
bash $ALPHAPOSE_SINGULARITY_DIR/scripts/setup_venv.sh

# Download model weights
bash $ALPHAPOSE_SINGULARITY_DIR/scripts/download_models.sh