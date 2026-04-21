#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TYPE=""
USE_SLURM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type) TYPE="$2"; shift 2 ;;
        --slurm) USE_SLURM=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TYPE" ]]; then
    echo "Usage: $0 --type <estimator> [--slurm]" >&2
    echo "Available types: openpose, mediapipe, alphapose, simplest_x, mmposewholebody, openpifpaf, sdpose" >&2
    exit 1
fi

SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then
    SLURM_ARG="--slurm"
fi

case "$TYPE" in
    openpose)
        bash "$SCRIPT_DIR/estimators/openpose/install_openpose.sh" $SLURM_ARG
        ;;
    mediapipe)
        bash "$SCRIPT_DIR/estimators/mediapipe/install_mediapipe.sh" $SLURM_ARG
        ;;
    openpifpaf)
        bash "$SCRIPT_DIR/estimators/openpifpaf/install_openpifpaf.sh" $SLURM_ARG
        ;;
    alphapose)
        bash "$SCRIPT_DIR/estimators/alphapose/install_alphapose.sh" $SLURM_ARG
        ;;
    mmposewholebody)
        bash "$SCRIPT_DIR/estimators/mmposewholebody/install_mmposewholebody.sh" $SLURM_ARG
        ;;
    simplest_x)
        bash "$SCRIPT_DIR/estimators/simplest_x/install_simplest_x.sh" $SLURM_ARG
        ;;
    sdpose)
        bash "$SCRIPT_DIR/estimators/sdpose/install_sdpose.sh" $SLURM_ARG
        ;;
    sapiens)
        bash "$SCRIPT_DIR/estimators/sapiens/install_sapiens.sh" $SLURM_ARG
        ;;
    *)
        echo "Unknown estimator type: $TYPE" >&2
        echo "Available types: openpose, mediapipe, alphapose, simplest_x, mmposewholebody, openpifpaf, sdpose, sapiens" >&2
        exit 1
        ;;
esac
