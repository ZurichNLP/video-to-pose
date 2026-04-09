#!/usr/bin/env bash
set -euo pipefail

SIMPLEST_X_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SIMPLEST_X_DIR")"
TOOLS=$REPO_DIR/tools

USE_SLURM=false
INPUT=""
OUTPUT=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)  USE_SLURM=true; shift ;;
        --input)  INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$DEVICE" == "cpu" ]]; then
    echo "Error: simplest_x does not support CPU. Use --device gpu or omit --device." >&2
    exit 1
fi

if [[ "$USE_SLURM" = true ]]; then
    echo "Error: --slurm is not yet supported for simplest_x." >&2
    exit 1
fi

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --input <input_folder> --output <output_folder> [--device gpu]" >&2
    exit 1
fi

SMPLEST_X_REPO=$TOOLS/simplest_x/SMPLest-X
VENV_DIR="$TOOLS/simplest_x/venv"

if [[ ! -d "$SMPLEST_X_REPO" ]]; then
    echo "Error: SMPLest-X repo not found. Please run install.sh --type simplest_x first." >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: Python venv not found. Please run install.sh --type simplest_x first." >&2
    exit 1
fi

if [[ ! -f "$SMPLEST_X_REPO/pretrained_models/smplest_x_h/smplest_x_h.pth.tar" ]]; then
    echo "Error: model checkpoint not found. Please run install.sh --type simplest_x first." >&2
    exit 1
fi

if [[ ! -d "$SMPLEST_X_REPO/human_models/smplx" ]]; then
    echo "Error: SMPL-X body models not found. Please run install.sh --type simplest_x first." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

JSON_TMP=$(mktemp -d)
trap "rm -rf $JSON_TMP" EXIT

# Resolve to absolute paths before cd-ing into the SMPLest-X repo, since
# json_pose_estimator.py uses relative paths (e.g. ./pretrained_models/yolov8x.pt)
# from the config and must be run from inside $SMPLEST_X_REPO.
INPUT_ABS="$(cd "$INPUT" && pwd)"

# Use a subshell to avoid changing the calling script's working directory.
(cd "$SMPLEST_X_REPO" && PYTHONPATH="$SMPLEST_X_REPO" python main/json_pose_estimator.py \
    --video_path "$INPUT_ABS" \
    --ckpt_name smplest_x_h \
    --json_output_path "$JSON_TMP")

mkdir -p "$OUTPUT"
for json_file in "$JSON_TMP"/*.json; do
    stem=$(basename "$json_file" .json)
    json_to_pose -i "$json_file" -o "$OUTPUT/${stem}.pose" --format smplest-x
done

deactivate
