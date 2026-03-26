# SMPLest-X

## Estimator-specific arguments

Additional arguments specific to SMPLest-X that can be passed directly to the main script:

`--device`: only `gpu` is supported. Passing `--device cpu` will fail with an error.

## Model and code details

The SMPLest-X-H (huge) model is used, producing **139 keypoints** in total across 4 components:

| Component             | Keypoints |
|-----------------------|-----------|
| Body joints           | 25        |
| Left hand joints      | 21        |
| Right hand joints     | 21        |
| Face landmarks        | 72        |
| **Total**             | **139**   |

Each keypoint has 2 coordinates (x, y) in image space. Only 1 person is detected per frame.

The scripts in this directory use the GerrySant fork of SMPLest-X
(https://github.com/GerrySant/SMPLest-X, branch `pose_estimation_study`), which adds
`json_pose_estimator.py` for machine-readable keypoint output. The original MotrixLab/SMPLest-X
only saves visualization images and does not support JSON export.

Pose conversion from JSON to `.pose` format uses the GerrySant fork of pose-format
(https://github.com/GerrySant/pose, branch `multiple_support`), which adds
`pose_format/utils/smplest_x.py` with `load_smplestx_pose()`.

## Requirements

- Python 3
- NVIDIA GPU with CUDA drivers (CPU mode is not supported)
- ~8.2 GB disk space for the model checkpoint, plus additional space for SMPL-X body models

## Cite

```bibtex
@article{yin2025smplest,
  title={SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation},
  author={Yin, Wanqi and Cai, Zhongang and Wang, Ruisi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Mei, Haiyi and Wang, Yanjun and Pang, Hui En and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yamashita, Atsushi and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2501.09782},
  year={2025}
}
```
