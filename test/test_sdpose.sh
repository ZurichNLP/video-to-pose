#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$SCRIPT_DIR/data/input"
OUTPUT_DIR="$SCRIPT_DIR/data/output/sdpose"

# Download test data if not already present
bash "$SCRIPT_DIR/download_test_data.sh"

# Parse command line arguments
USE_SLURM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM="--slurm"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Install sdpose
bash "$REPO_DIR/install.sh" --type sdpose $USE_SLURM

# Run pose estimation
mkdir -p "$OUTPUT_DIR"
bash "$REPO_DIR/videos_to_poses.sh" \
    --type sdpose \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR"

SDPOSE_VENV="$REPO_DIR/estimators/tools/sdpose/venv"

# Test output pose shape
source "$SDPOSE_VENV/bin/activate"
pytest "$SCRIPT_DIR/test_pose_shape.py::test_sdpose_shape"
deactivate

echo "Test passed."
