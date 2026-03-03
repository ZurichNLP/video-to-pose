#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TEST_VIDEO_URL="https://www.sgb-fss.ch/signsuisse/fileadmin/signsuisse_ressources/videos/262C723C-021E-5B4E-C607E2CE094D1963.mp4"
DOWNLOADED_VIDEO="$SCRIPT_DIR/data/test_video.mp4"
INPUT_DIR="$SCRIPT_DIR/data/input"

if [[ ! -f "$DOWNLOADED_VIDEO" ]]; then
    mkdir -p "$(dirname "$DOWNLOADED_VIDEO")"
    wget -O "$DOWNLOADED_VIDEO" "$TEST_VIDEO_URL"
fi

echo "Test video: $DOWNLOADED_VIDEO ($(du -sh "$DOWNLOADED_VIDEO" | cut -f1))"

mkdir -p "$INPUT_DIR"

for i in 1 2 3; do
    cp "$DOWNLOADED_VIDEO" "$INPUT_DIR/test_video_$i.mp4"
done

echo "Input folder: $INPUT_DIR"
echo "Files:"
ls -lh "$INPUT_DIR"