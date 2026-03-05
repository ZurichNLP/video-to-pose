import glob
import os

import pytest
from pose_format import Pose

TEST_DIR = os.path.dirname(__file__)

# ── OpenPose ──────────────────────────────────────────────────────────────────

OPENPOSE_OUTPUT_DIR = os.path.join(TEST_DIR, "data", "output", "openpose")
OPENPOSE_NUM_KEYPOINTS = 137  # 25 body + 70 face + 21 left hand + 21 right hand
OPENPOSE_NUM_COORDS = 2  # x, y


def get_openpose_files():
    return glob.glob(os.path.join(OPENPOSE_OUTPUT_DIR, "**", "*.pose"), recursive=True)


@pytest.mark.parametrize("pose_file", get_openpose_files())
def test_openpose_shape(pose_file):
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    # shape: (frames, people, keypoints, coordinates)
    shape = pose.body.data.shape

    assert len(shape) == 4, f"Expected 4 dimensions, got {len(shape)}"

    frames, people, keypoints, coords = shape

    assert frames == 62, f"Expected 62 frames, got {frames}"
    assert people >= 1, "Expected at least one person"
    assert keypoints == OPENPOSE_NUM_KEYPOINTS, (
        f"Expected {OPENPOSE_NUM_KEYPOINTS} keypoints (OpenPose 137-keypoint model), got {keypoints}"
    )
    assert coords == OPENPOSE_NUM_COORDS, (
        f"Expected {OPENPOSE_NUM_COORDS} coordinates (x, y), got {coords}"
    )


# ── MediaPipe ─────────────────────────────────────────────────────────────────

MEDIAPIPE_OUTPUT_DIR = os.path.join(TEST_DIR, "data", "output", "mediapipe")
# 33 pose + 478 face (with iris refinement) + 21 left hand + 21 right hand + 33 pose world
MEDIAPIPE_NUM_KEYPOINTS = 586
MEDIAPIPE_NUM_COORDS = 3  # x, y, z


def get_mediapipe_files():
    return glob.glob(os.path.join(MEDIAPIPE_OUTPUT_DIR, "**", "*.pose"), recursive=True)


@pytest.mark.parametrize("pose_file", get_mediapipe_files())
def test_mediapipe_shape(pose_file):
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    # shape: (frames, people, keypoints, coordinates)
    shape = pose.body.data.shape

    assert len(shape) == 4, f"Expected 4 dimensions, got {len(shape)}"

    frames, people, keypoints, coords = shape

    assert frames == 62, f"Expected 62 frames, got {frames}"
    assert people == 1, "MediaPipe Holistic detects exactly 1 person"
    assert keypoints == MEDIAPIPE_NUM_KEYPOINTS, (
        f"Expected {MEDIAPIPE_NUM_KEYPOINTS} keypoints (MediaPipe Holistic with iris refinement), got {keypoints}"
    )
    assert coords == MEDIAPIPE_NUM_COORDS, (
        f"Expected {MEDIAPIPE_NUM_COORDS} coordinates (x, y, z), got {coords}"
    )


# ── OpenPifPaf ────────────────────────────────────────────────────────────────

OPENPIFPAF_OUTPUT_DIR = os.path.join(TEST_DIR, "data", "output", "openpifpaf")
# 133 keypoints: 17 body + 68 face + 21 left hand + 21 right hand + 6 foot
OPENPIFPAF_NUM_KEYPOINTS = 133
OPENPIFPAF_NUM_COORDS = 2  # x, y


def get_openpifpaf_files():
    return glob.glob(os.path.join(OPENPIFPAF_OUTPUT_DIR, "**", "*.pose"), recursive=True)


@pytest.mark.parametrize("pose_file", get_openpifpaf_files())
def test_openpifpaf_shape(pose_file):
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    # shape: (frames, people, keypoints, coordinates)
    # Note: frames where no person is detected are omitted, so frame count
    # may be less than the total video frame count.
    shape = pose.body.data.shape

    assert len(shape) == 4, f"Expected 4 dimensions, got {len(shape)}"

    frames, people, keypoints, coords = shape

    assert frames >= 1, "Expected at least one frame with a detection"
    assert people >= 1, "Expected at least one person"
    assert keypoints == OPENPIFPAF_NUM_KEYPOINTS, (
        f"Expected {OPENPIFPAF_NUM_KEYPOINTS} keypoints (OpenPifPaf wholebody 133-keypoint model), got {keypoints}"
    )
    assert coords == OPENPIFPAF_NUM_COORDS, (
        f"Expected {OPENPIFPAF_NUM_COORDS} coordinates (x, y), got {coords}"
    )