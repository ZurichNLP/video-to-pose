#!/usr/bin/env bash
set -euo pipefail

OPENPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
ESTIMATORS_DIR="$(dirname "$OPENPOSE_DIR")"
REPO_DIR="$(dirname "$ESTIMATORS_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
CHUNKS=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)  USE_SLURM=true; shift ;;
        --input)  INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$DEVICE" == "cpu" ]]; then
    echo "Error: openpose does not support CPU. Use --device gpu or omit --device." >&2
    exit 1
fi

if [[ -n "$CHUNKS" && "$USE_SLURM" = false ]]; then
    echo "--chunks requires --slurm" >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--slurm] [--chunks N]" >&2
    exit 1
fi

OPENPOSE_SINGULARITY_DIR=$TOOLS/openpose/openpose-singularity-uzh

if [[ ! -d "$OPENPOSE_SINGULARITY_DIR" ]]; then
    echo "Error: openpose-singularity-uzh repo not found. Please run install.sh --type openpose first." >&2
    exit 1
fi

if [[ ! -f "$OPENPOSE_SINGULARITY_DIR/openpose.sif" ]]; then
    echo "Error: openpose.sif not found. Please run install.sh --type openpose first (or wait for the SLURM build job to complete)." >&2
    exit 1
fi

if [[ ! -d "$OPENPOSE_SINGULARITY_DIR/venv" ]]; then
    echo "Error: Python venv not found. Please run install.sh --type openpose first." >&2
    exit 1
fi

if [ "$USE_SLURM" = true ]; then
    CHUNKS_ARG=""
    if [[ -n "$CHUNKS" ]]; then
        CHUNKS_ARG="--chunks $CHUNKS"
    fi
    bash $OPENPOSE_SINGULARITY_DIR/scripts/slurm_submit.sh "$INPUT" "$OUTPUT" $CHUNKS_ARG
else
    bash $OPENPOSE_SINGULARITY_DIR/scripts/batch_to_pose.sh "$INPUT" "$OUTPUT"
fi
