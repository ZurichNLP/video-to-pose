# MMPose WholeBody

## Estimator-specific arguments

`--device cpu|gpu`: device to use for inference (default: `cpu` on macOS, `gpu` elsewhere). CPU is slow; use only if no GPU is available.

## Model and code details

Uses the MMPose `wholebody` model alias, which runs a top-down whole-body pose estimator producing **133 keypoints** across 4 components:

| Component    | Keypoints |
|--------------|-----------|
| Body         | 23        |
| Face         | 68        |
| Left hand    | 21        |
| Right hand   | 21        |
| **Total**    | **133**   |

Each keypoint has X, Y coordinates plus a confidence score. While MMPose WholeBody does support the detection of multiple people per frame, please note that when multiple people are detected, our implementation takes only the person with the highest confidence score. 

Pose estimation uses [`MMPoseInferencer`](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html) with the `wholebody` model alias. Output is written in `.pose` format using a COCO-WholeBody 133 header from the `pose-format` library.

## Requirements

- Python 3 (Python 3.12 is recommended for package version compatibility)
- GPU with CUDA strongly recommended (CPU inference is very slow)

## Installation notes for CPU / GPU

When the `.install_mmposewholebody.sh` script detects GPU in the environment, it automatically runs `module load cuda/12.6.3`. If this syntax is not compatible with loading cuda in your environment, you must load cuda before running the install script. 

Furthermore, the `.install_mmposewholebody.sh` script installs different `pytorch` and `torchvision` versions depending on whether or not GPU is available in the environment. If you have GPU in your environment but wish to run MMPose WholeBody with CPU, you should manually downgrade these packages. 

## Cite

```bibtex
@inproceedings{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
