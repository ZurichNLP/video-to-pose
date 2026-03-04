# MMPose WholeBody

## Estimator-specific arguments

`--use-cpu`: run inference on CPU instead of GPU (slow; use only if no GPU is available).

## Model and code details

Uses the MMPose `wholebody` model alias, which runs a top-down whole-body pose estimator producing **133 keypoints** across 4 components:

| Component    | Keypoints |
|--------------|-----------|
| Body         | 23        |
| Face         | 68        |
| Left hand    | 21        |
| Right hand   | 21        |
| **Total**    | **133**   |

Each keypoint has X, Y coordinates plus a confidence score. Multiple people per frame are supported.

Pose estimation uses [`MMPoseInferencer`](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html) with the `wholebody` model alias. Output is written in `.pose` format using a COCO-WholeBody 133 header from the `pose-format` library.

## Requirements

- Python 3
- GPU with CUDA strongly recommended (CPU inference is very slow)

The install script installs a CPU-only PyTorch build by default. For GPU inference, reinstall PyTorch for your CUDA version before or after running `install.sh`:
```
https://pytorch.org/get-started/locally/
```

## Cite

```bibtex
@inproceedings{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
