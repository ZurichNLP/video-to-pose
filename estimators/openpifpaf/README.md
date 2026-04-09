# OpenPifPaf

## Estimator-specific arguments

## Model and code details

This will run pose estimation using [OpenPifPaf](https://openpifpaf.github.io/) with the
`shufflenetv2k30-wholebody` checkpoint, producing 133-keypoint COCO-WholeBody
output (same format as mmposewholebody).

OpenPifPaf outputs 133 keypoints in the **COCO-Wholebody-133** format. 

| Component           | Keypoints |
|---------------------|-----------|
| Body landmarks      | 17        |
| Face landmarks      | 68        |
| Left hand landmarks | 21        |
| Right hand landmarks| 21        |
| Foot landmarks      | 6         |
| **Total**           | **133**   |

Output shape: `(frames, people, 133, 2)` and a separate `confidence` array with the shape shape `(frames, people, 133)`. In this repo, these outputs are converted into the `.pose` format. 

The model checkpoint (`shufflenetv2k30-wholebody`, ~100 MB) is downloaded automatically on first run and cached in `~/.cache/torch/hub/checkpoints/`.

Pose estimation is performed via the `videos_to_poses` command from a fork of the 
[`pose-format`](https://github.com/sign-language-processing/pose) library located [`here`](https://github.com/catherine-o-brien/pose/tree/new_estimators), which is a wrapper around the OpenPifPaf implementation. 

In its original implementation, OpenPifPaf can detect multiple people per frame. In this repo, only the detected person with the highest confidence values is included in the output– so `people` will never be more than 1. Please note that frames where no person is detected are omitted from the output, so the outputted frame count may be less than the total number of video frames.

## Requirements
- GPU is preferred but optional
- Due to its dependency on `torchvision<0.15`, OpenPifPaf is only compatible with `Python=3.9` and `Python=3.10`. To accommodate on this on the SLURM cluster, which only has `python=3.12`, we create a conda environment with `python=3.10` and create the venv with `python=3.12` within that conda environment. 
- When the `install.sh` script detects GPU, CUDA is loaded with `module load cuda/12.6.3`. If this is incorrect syntax on your system, you should load CUDA manually prior to running the install script. 

## Cite
```bibtex
@article{kreiss2021openpifpaf,
  title={Openpifpaf: Composite fields for semantic keypoint detection and spatio-temporal association},
  author={Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={8},
  pages={13498--13511},
  year={2021},
  publisher={IEEE}
}
```
