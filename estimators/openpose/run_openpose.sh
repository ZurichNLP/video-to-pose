#!/usr/bin/env bash
set -euo pipefail

OPENPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$OPENPOSE_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
CHUNKS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm) USE_SLURM=true; shift ;;
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$CHUNKS" && "$USE_SLURM" = false ]]; then
    echo "--chunks requires --slurm" >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--slurm] [--chunks N]" >&2
    exit 1
fi

OPENPOSE_SINGULARITY_DIR=$TOOLS/openpose/openpose-singularity-uzh

if [ "$USE_SLURM" = true ]; then
    CHUNKS_ARG=""
    if [[ -n "$CHUNKS" ]]; then
        CHUNKS_ARG="--chunks $CHUNKS"
    fi
    bash $OPENPOSE_SINGULARITY_DIR/scripts/slurm_submit.sh "$INPUT" "$OUTPUT" $CHUNKS_ARG
else
    bash $OPENPOSE_SINGULARITY_DIR/scripts/batch_to_pose.sh "$INPUT" "$OUTPUT"
fi
