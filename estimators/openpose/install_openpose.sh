#!/usr/bin/env bash
set -euo pipefail

OPENPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$OPENPOSE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false

for arg in "$@"; do
    case "$arg" in
        --slurm) USE_SLURM=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p $TOOLS/openpose

OPENPOSE_SINGULARITY_DIR=$TOOLS/openpose/openpose-singularity-uzh

if [[ ! -d "$OPENPOSE_SINGULARITY_DIR" ]]; then
    git clone https://github.com/bricksdont/openpose-singularity-uzh $OPENPOSE_SINGULARITY_DIR
fi

# steps taken from here: https://github.com/bricksdont/openpose-singularity-uzh/tree/main

# 1. Build (pull) the Singularity container image (~10-15 min)
# Login nodes may not have enough memory; use --slurm to submit as a SLURM job instead.
if [ "$USE_SLURM" = true ]; then
    # slurm_build_container.sh uses $SLURM_SUBMIT_DIR to locate scripts/, so sbatch
    # must be called from the repo root.
    (cd $OPENPOSE_SINGULARITY_DIR && sbatch scripts/slurm_build_container.sh)
    echo "Container build submitted as SLURM job. Monitor with: squeue -u \$USER"
else
    bash $OPENPOSE_SINGULARITY_DIR/scripts/build_container.sh
fi

# 5. Set up Python virtual environment
bash $OPENPOSE_SINGULARITY_DIR/scripts/setup_venv.sh
