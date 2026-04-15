#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TYPE=""
INPUT=""
OUTPUT=""
DEVICE=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)   TYPE="$2";   shift 2 ;;
        --input)  INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
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
    echo "Usage: $0 --type <estimator> --input <input_folder> --output <output_folder> [--device cpu|gpu] [--slurm] [...]" >&2
    echo "Available types: openpose, mediapipe, alphapose, simplest_x, mmposewholebody, openpifpaf, sdpose" >&2
    exit 1
fi

if [[ -n "$DEVICE" && "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: --device must be 'cpu' or 'gpu', got '$DEVICE'." >&2
    exit 1
fi

DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi

case "$TYPE" in
    openpose)
        bash $SCRIPT_DIR/estimators/openpose/run_openpose.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    mediapipe)
        bash $SCRIPT_DIR/estimators/mediapipe/run_mediapipe.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    alphapose)
        bash $SCRIPT_DIR/estimators/alphapose/run_alphapose.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    mmposewholebody)
        bash $SCRIPT_DIR/estimators/mmposewholebody/run_mmposewholebody.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    openpifpaf)
        bash $SCRIPT_DIR/estimators/openpifpaf/run_openpifpaf.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    simplest_x)
        bash $SCRIPT_DIR/estimators/simplest_x/run_simplest_x.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${DEVICE_ARG[@]}" \
            "${PASSTHROUGH[@]}"
        ;;
    sdpose)
        bash $SCRIPT_DIR/estimators/sdpose/run_sdpose.sh \
            --input "$INPUT" \
            --output "$OUTPUT" \
            "${PASSTHROUGH[@]}"
        ;;
    *)
        echo "Unknown estimator type: $TYPE" >&2
        echo "Available types: openpose, mediapipe, alphapose, simplest_x, mmposewholebody, openpifpaf, sdpose" >&2
        exit 1
        ;;
esac