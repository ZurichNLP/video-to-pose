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
    echo "Available types: openpose, mediapipe, openpifpaf" >&2
    exit 1
fi

SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then
    SLURM_ARG="--slurm"
fi

case "$TYPE" in
    openpose)
        bash $SCRIPT_DIR/estimators/openpose/install_openpose.sh $SLURM_ARG
        ;;
    mediapipe)
        bash $SCRIPT_DIR/estimators/mediapipe/install_mediapipe.sh $SLURM_ARG
        ;;
    openpifpaf)
        bash $SCRIPT_DIR/estimators/openpifpaf/install_openpifpaf.sh $SLURM_ARG
        ;;
    *)
        echo "Unknown estimator type: $TYPE" >&2
        echo "Available types: openpose, mediapipe, openpifpaf" >&2
        exit 1
        ;;
esac
