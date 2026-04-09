#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

MAIN="$REPO_DIR/videos_to_poses.sh"

assert_fails() {
    local description="$1"
    shift
    if bash "$@" 2>/dev/null; then
        echo "FAIL: expected failure but succeeded: $description" >&2
        exit 1
    fi
    echo "OK: $description"
}

# Invalid --device value is rejected at the main script level
assert_fails "--device with invalid value 'cuda'" \
    "$MAIN" --type mediapipe --input /tmp --output /tmp --device cuda

# GPU-only estimators reject --device cpu
assert_fails "openpose rejects --device cpu" \
    "$MAIN" --type openpose --input /tmp --output /tmp --device cpu

assert_fails "alphapose rejects --device cpu" \
    "$MAIN" --type alphapose --input /tmp --output /tmp --device cpu

# CPU-only estimators reject --device gpu
assert_fails "mediapipe rejects --device gpu" \
    "$MAIN" --type mediapipe --input /tmp --output /tmp --device gpu

# GPU-only estimators (continued)
assert_fails "simplest_x rejects --device cpu" \
    "$MAIN" --type simplest_x --input /tmp --output /tmp --device cpu

echo "All argument validation tests passed."