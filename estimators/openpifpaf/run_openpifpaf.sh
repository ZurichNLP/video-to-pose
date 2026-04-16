#!/usr/bin/env bash
set -euo pipefail

OPENPIFPAF_DIR="$(cd "$(dirname "$0")" && pwd)"
ESTIMATORS_DIR="$(dirname "$OPENPIFPAF_DIR")"
REPO_DIR="$(dirname "$ESTIMATORS_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$DEVICE" && "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: --device must be 'cpu' or 'gpu'." >&2
    exit 1
fi

if [ "$USE_SLURM" = true ]; then
    echo "Error: --slurm is not yet supported for openpifpaf." >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--device cpu|gpu]" >&2
    exit 1
fi

VENV_DIR="$TOOLS/openpifpaf/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: openpifpaf venv not found. Please run install.sh --type openpifpaf first." >&2
    exit 1
fi

# videos_to_poses writes .pose files alongside input videos, so we run it on
# the input directory and then move the results to the output directory.
source "$VENV_DIR/bin/activate"

USE_CPU_ARG=""
if [ "$DEVICE" = "cpu" ]; then
    USE_CPU_ARG="--use-cpu"
fi

videos_to_poses \
    --format openpifpaf \
    --directory "$INPUT" \
    $USE_CPU_ARG
deactivate

mkdir -p "$OUTPUT"
find "$INPUT" -maxdepth 1 -name "*.pose" -exec mv {} "$OUTPUT"/ \;
