import glob
import os

import pytest
from pose_format import Pose

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "output", "openpose")

OPENPOSE_NUM_KEYPOINTS = 137  # 25 body + 70 face + 21 left hand + 21 right hand
OPENPOSE_NUM_COORDS = 2  # x, y


def get_pose_files():
    return glob.glob(os.path.join(OUTPUT_DIR, "**", "*.pose"), recursive=True)


@pytest.mark.parametrize("pose_file", get_pose_files())
def test_openpose_shape(pose_file):
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())

    # shape: (frames, people, keypoints, coordinates)
    shape = pose.body.data.shape

    assert len(shape) == 4, f"Expected 4 dimensions, got {len(shape)}"

    frames, people, keypoints, coords = shape

    assert frames == 133, f"Expected 133 frames, got {frames}"
    assert people >= 1, "Expected at least one person"
    assert keypoints == OPENPOSE_NUM_KEYPOINTS, (
        f"Expected {OPENPOSE_NUM_KEYPOINTS} keypoints (OpenPose 137-keypoint model), got {keypoints}"
    )
    assert coords == OPENPOSE_NUM_COORDS, (
        f"Expected {OPENPOSE_NUM_COORDS} coordinates (x, y), got {coords}"
    )