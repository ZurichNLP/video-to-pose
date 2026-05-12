#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

USE_SLURM=false
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --*)
            PASSTHROUGH+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                PASSTHROUGH+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then SLURM_ARG="--slurm"; fi

INPUT_DIR="$SCRIPT_DIR/data/input"
OUTPUT_DIR="$SCRIPT_DIR/data/output/sapiens"

# Download test data if not already present
bash "$SCRIPT_DIR/download_test_data.sh"

# Install sapiens
bash "$REPO_DIR/install.sh" --type sapiens $SLURM_ARG

# Run pose estimation
mkdir -p "$OUTPUT_DIR"
bash "$REPO_DIR/videos_to_poses.sh" \
    --type sapiens \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    $SLURM_ARG \
    "${PASSTHROUGH[@]}"

SAPIENS_VENV="$REPO_DIR/tools/sapiens/venv"

# Test output pose shape
source "$SAPIENS_VENV/bin/activate"
pytest "$SCRIPT_DIR/test_pose_shape.py::test_sapiens_shape"
deactivate

echo "Test passed."
