#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$SCRIPT_DIR/data/input"
OUTPUT_DIR="$SCRIPT_DIR/data/output/simplest_x"

# Download test data if not already present
bash "$SCRIPT_DIR/download_test_data.sh"

# Install simplest_x
bash "$REPO_DIR/install.sh" --type simplest_x

# Run pose estimation
mkdir -p "$OUTPUT_DIR"
bash "$REPO_DIR/videos_to_poses.sh" \
    --type simplest_x \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR"

SIMPLEST_X_VENV="$REPO_DIR/tools/simplest_x/venv"

# Test output pose shape
source "$SIMPLEST_X_VENV/bin/activate"
pytest "$SCRIPT_DIR/test_pose_shape.py::test_simplest_x_shape"
deactivate

echo "Test passed."
