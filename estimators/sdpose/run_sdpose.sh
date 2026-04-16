#!/usr/bin/env bash
set -euo pipefail

SDPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SDPOSE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
DEVICE=""
NUM_WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$DEVICE" && "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: --device must be 'cpu' or 'gpu'." >&2
    exit 1
fi

if [ "$USE_SLURM" = true ]; then
    echo "Error: --slurm is not yet supported for sdpose." >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--device cpu|gpu]" >&2
    exit 1
fi

VENV_DIR="$TOOLS/sdpose/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: sdpose venv not found. Please run install.sh --type sdpose first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
NUM_WORKERS_ARG=""
if [[ -n "$NUM_WORKERS" ]]; then
    NUM_WORKERS_ARG="--num-workers $NUM_WORKERS"
fi
USE_CPU_ARG=""
if [[ "$DEVICE" == "cpu" ]]; then
    USE_CPU_ARG="--use-cpu"
fi

videos_to_poses \
    --format sdpose \
    --directory "$INPUT" \
    $NUM_WORKERS_ARG \
    $USE_CPU_ARG
deactivate

mkdir -p "$OUTPUT"
find "$INPUT" -maxdepth 1 -name "*.pose" -exec mv {} "$OUTPUT"/ \;
