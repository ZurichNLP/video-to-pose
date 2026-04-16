#!/usr/bin/env bash
set -euo pipefail

ALPHAPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
ESTIMATORS_DIR="$(dirname "$ALPHAPOSE_DIR")"
REPO_DIR="$(dirname "$ESTIMATORS_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
CHUNKS=""
KEYPOINTS=""
LOWPRIO=false
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)     USE_SLURM=true; shift ;;
        --input)     INPUT="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --chunks)    CHUNKS="$2"; shift 2 ;;
        --keypoints) KEYPOINTS="$2"; shift 2 ;;
        --lowprio)   LOWPRIO=true; shift ;;
        --device)    DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$DEVICE" == "cpu" ]]; then
    echo "Error: alphapose does not support CPU. Use --device gpu or omit --device." >&2
    exit 1
fi

if [[ -n "$CHUNKS" && "$USE_SLURM" = false ]]; then
    echo "--chunks requires --slurm" >&2
    exit 1
fi

if [[ "$LOWPRIO" = true && "$USE_SLURM" = false ]]; then
    echo "--lowprio requires --slurm" >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--slurm] [--chunks N] [--keypoints 136|133] [--lowprio]" >&2
    exit 1
fi

ALPHAPOSE_SINGULARITY_DIR=$TOOLS/alphapose/alphapose-singularity-uzh

if [[ ! -d "$ALPHAPOSE_SINGULARITY_DIR" ]]; then
    echo "Error: alphapose-singularity-uzh repo not found. Please run install.sh --type alphapose first." >&2
    exit 1
fi

if [[ ! -f "$ALPHAPOSE_SINGULARITY_DIR/alphapose.sif" ]]; then
    echo "Error: alphapose.sif not found. Please run install.sh --type alphapose first (or wait for the SLURM build job to complete)." >&2
    exit 1
fi

if [[ ! -d "$ALPHAPOSE_SINGULARITY_DIR/venv" ]]; then
    echo "Error: Python venv not found. Please run install.sh --type alphapose first." >&2
    exit 1
fi

if [[ ! -d "$ALPHAPOSE_SINGULARITY_DIR/data/models" ]]; then
    echo "Error: model weights not found. Please run install.sh --type alphapose first." >&2
    exit 1
fi

KEYPOINTS_ARG=""
if [[ -n "$KEYPOINTS" ]]; then
    KEYPOINTS_ARG="--keypoints $KEYPOINTS"
fi

if [ "$USE_SLURM" = true ]; then
    CHUNKS_ARG=""
    if [[ -n "$CHUNKS" ]]; then
        CHUNKS_ARG="--chunks $CHUNKS"
    fi
    LOWPRIO_ARG=""
    if [[ "$LOWPRIO" = true ]]; then
        LOWPRIO_ARG="--lowprio"
    fi
    bash $ALPHAPOSE_SINGULARITY_DIR/scripts/slurm_submit.sh "$INPUT" "$OUTPUT" $CHUNKS_ARG $KEYPOINTS_ARG $LOWPRIO_ARG
else
    bash $ALPHAPOSE_SINGULARITY_DIR/scripts/batch_to_pose.sh "$INPUT" "$OUTPUT" $KEYPOINTS_ARG
fi