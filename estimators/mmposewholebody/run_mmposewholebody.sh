#!/usr/bin/env bash
set -euo pipefail

MMPOSEWHOLEBODY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$MMPOSEWHOLEBODY_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
# Default to CPU on macOS: the installed torch build includes MPS (Apple Silicon
# GPU), but PyTorch 2.1.0's MPS backend has a known bug with sort(descending=True)
# that breaks mmdet inference. Force CPU unless the caller explicitly opts out
# with --no-cpu.
if [[ "$(uname -s)" == "Darwin" ]]; then
    USE_CPU=true
else
    USE_CPU=false
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --use-cpu) USE_CPU=true; shift ;;
        --no-cpu) USE_CPU=false; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "$USE_SLURM" = true ]; then
    echo "Error: --slurm is not yet supported for mmposewholebody." >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--use-cpu]" >&2
    exit 1
fi

VENV_DIR="$TOOLS/mmposewholebody/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: mmposewholebody venv not found. Please run install.sh --type mmposewholebody first." >&2
    exit 1
fi

# videos_to_poses writes .pose files alongside input videos, so we run it on
# the input directory and then move the results to the output directory.
source "$VENV_DIR/bin/activate"
NUM_WORKERS_ARG=""
if [[ -n "${NUM_WORKERS:-}" ]]; then
    NUM_WORKERS_ARG="--num-workers $NUM_WORKERS"
fi
USE_CPU_ARG=""
if [ "$USE_CPU" = true ]; then
    USE_CPU_ARG="--use-cpu"
fi

videos_to_poses \
    --format mmposewholebody \
    --directory "$INPUT" \
    $NUM_WORKERS_ARG \
    $USE_CPU_ARG
deactivate

mkdir -p "$OUTPUT"
find "$INPUT" -maxdepth 1 -name "*.pose" -exec mv {} "$OUTPUT"/ \;