#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TYPE=""
INPUT=""
OUTPUT=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)   TYPE="$2";   shift 2 ;;
        --input)  INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
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

if [[ -z "$TYPE" || -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --type <estimator> --input <input_folder> --output <output_folder> [--slurm] [...]" >&2
    echo "Available types: openpose, mediapipe, mmposewholebody" >&2
    exit 1
fi

case "$TYPE" in
    openpose)
        bash $SCRIPT_DIR/estimators/openpose/run_openpose.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${PASSTHROUGH[@]}"
        ;;
    mediapipe)
        bash $SCRIPT_DIR/estimators/mediapipe/run_mediapipe.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${PASSTHROUGH[@]}"
        ;;
    mmposewholebody)
        bash $SCRIPT_DIR/estimators/mmposewholebody/run_mmposewholebody.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${PASSTHROUGH[@]}"
        ;;
    *)
        echo "Unknown estimator type: $TYPE" >&2
        echo "Available types: openpose, mediapipe, mmposewholebody" >&2
        exit 1
        ;;
esac