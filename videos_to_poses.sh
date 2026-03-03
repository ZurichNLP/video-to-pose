#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TYPE=""
INPUT=""
OUTPUT=""
USE_SLURM=false
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)  TYPE="$2";   shift 2 ;;
        --input) INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --slurm) USE_SLURM=true; shift ;;
        --extra) read -ra EXTRA <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TYPE" || -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --type <estimator> --input <input_folder> --output <output_folder> [--slurm] [--extra <extra_args>]" >&2
    echo "Available types: openpose, mediapipe" >&2
    exit 1
fi

SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then
    SLURM_ARG="--slurm"
fi

case "$TYPE" in
    openpose)
        bash $SCRIPT_DIR/estimators/openpose/run_openpose.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            $SLURM_ARG \
            "${EXTRA[@]}"
        ;;
    mediapipe)
        bash $SCRIPT_DIR/estimators/mediapipe/run_mediapipe.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            $SLURM_ARG \
            "${EXTRA[@]}"
        ;;
    *)
        echo "Unknown estimator type: $TYPE" >&2
        echo "Available types: openpose, mediapipe" >&2
        exit 1
        ;;
esac
