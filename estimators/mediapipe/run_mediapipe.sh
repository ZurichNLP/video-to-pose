#!/usr/bin/env bash
set -euo pipefail

MEDIAPIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MEDIAPIPE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
NUM_WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "$USE_SLURM" = true ]; then
    echo "Error: --slurm is not yet supported for mediapipe." >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--num-workers N]" >&2
    exit 1
fi

VENV_DIR="$TOOLS/mediapipe/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: mediapipe venv not found. Please run install.sh --type mediapipe first." >&2
    exit 1
fi

# videos_to_poses writes .pose files alongside input videos, so we run it on
# the input directory and then move the results to the output directory.
source "$VENV_DIR/bin/activate"
NUM_WORKERS_ARG=""
if [[ -n "$NUM_WORKERS" ]]; then
    NUM_WORKERS_ARG="--num-workers $NUM_WORKERS"
fi

videos_to_poses \
    --format mediapipe \
    --directory "$INPUT" \
    --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true" \
    $NUM_WORKERS_ARG
deactivate

mkdir -p "$OUTPUT"
find "$INPUT" -maxdepth 1 -name "*.pose" -exec mv {} "$OUTPUT"/ \;