#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$SCRIPT_DIR/data/input"
OUTPUT_DIR="$SCRIPT_DIR/data/output/mmposewholebody"

# Download test data if not already present
bash "$SCRIPT_DIR/download_test_data.sh"

# Install mmposewholebody
bash "$REPO_DIR/install.sh" --type mmposewholebody

# Run pose estimation
mkdir -p "$OUTPUT_DIR"
bash "$REPO_DIR/videos_to_poses.sh" \
    --type mmposewholebody \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR"

MMPOSEWHOLEBODY_VENV="$REPO_DIR/estimators/tools/mmposewholebody/venv"

# Test output pose shape
source "$MMPOSEWHOLEBODY_VENV/bin/activate"
pytest "$SCRIPT_DIR/test_pose_shape.py::test_mmposewholebody_shape"
deactivate

echo "Test passed."
