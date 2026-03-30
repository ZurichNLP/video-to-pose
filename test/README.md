# Tests

## Overview

Tests are organised per estimator. Each estimator has a dedicated shell script (`test_<name>.sh`) that:

1. Downloads shared test input data (if not already present)
2. Installs the estimator via the top-level `install.sh`
3. Runs pose estimation via the top-level `videos_to_poses.sh`
4. Activates the estimator's Python venv and runs pytest to validate the output

The Github CI will only run tests that do not require GPU access, so in that case some estimators will
be skipped.

## Test data

Test data is managed by `download_test_data.sh`, which downloads a single test video and copies it three times into `data/input/`:

```
data/
  input/
    test_video_1.mp4
    test_video_2.mp4
    test_video_3.mp4
```

The original download is cached at `data/test_video.mp4` so subsequent runs skip the download. The `data/` folder is gitignored.

Estimator output is written to `data/output/<estimator_name>/`, keeping results from different estimators separate.

## Running the tests

### OpenPose

**Requirements:** Singularity CE >= 3.x or Apptainer >= 1.x, NVIDIA GPU with CUDA drivers.

```bash
bash test/test_openpose.sh
```

This will:
- Download test videos (first run only)
- Clone the [openpose-singularity-uzh](https://github.com/bricksdont/openpose-singularity-uzh) repo and build the Singularity container (~10-15 min, first run only)
- Set up a Python venv (first run only)
- Run batch pose estimation on the three test videos
- Run the estimator-specific tests in `test/test_pose_shape.py` to validate the output

### AlphaPose

**Requirements:** Singularity CE >= 3.x or Apptainer >= 1.x, NVIDIA GPU with CUDA drivers. Not run on GitHub CI.

```bash
bash test/test_alphapose.sh
```

This will:
- Download test videos (first run only)
- Clone the [alphapose-singularity-uzh](https://github.com/bricksdont/alphapose-singularity-uzh) repo, pull the Singularity container (~8 GB, first run only), set up a Python venv, and download model weights (first run only)
- Run batch pose estimation on the three test videos
- Run the estimator-specific tests in `test/test_pose_shape.py` to validate the output

### MediaPipe

**Requirements:** Python 3.

```bash
bash test/test_mediapipe.sh
```

This will:
- Download test videos (first run only)
- Set up a Python venv with `mediapipe` and `pose-format` (first run only)
- Run batch pose estimation on the three test videos
- Run the estimator-specific tests in `test/test_pose_shape.py` to validate the output

## Output shape tests

`test_pose_shape.py` is a pytest file that loads every `.pose` file found under `data/output/<estimator>/` and asserts its shape is correct. It contains separate test functions per estimator.

### OpenPose

Expected shape of `pose.body.data` — `(frames, people, keypoints, coordinates)`:

| Dimension   | Expected value | Notes                                             |
|-------------|----------------|---------------------------------------------------|
| frames      | 62             | Fixed for the test video                          |
| people      | >= 1           | At least one person detected                      |
| keypoints   | 137            | 25 body + 70 face + 21 left hand + 21 right hand |
| coordinates | 2              | x, y                                              |

### AlphaPose

Expected shape of `pose.body.data` — `(frames, people, keypoints, coordinates)`:

| Dimension   | Expected value | Notes                        |
|-------------|----------------|------------------------------|
| frames      | 62             | Fixed for the test video     |
| people      | >= 1           | At least one person detected |
| keypoints   | 136            | HALPE_136 whole-body model   |
| coordinates | 2              | x, y                         |

### MediaPipe

Expected shape of `pose.body.data` — `(frames, people, keypoints, coordinates)`:

| Dimension   | Expected value | Notes                                                                        |
|-------------|----------------|------------------------------------------------------------------------------|
| frames      | 62             | Fixed for the test video                                                     |
| people      | 1              | MediaPipe Holistic detects exactly 1 person                                  |
| keypoints   | 586            | 33 pose + 478 face (with iris) + 21 left hand + 21 right hand + 33 pose world |
| coordinates | 3              | x, y, z                                                                      |