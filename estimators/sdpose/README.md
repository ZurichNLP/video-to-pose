# SDPose

## Estimator-specific arguments

Additional arguments specific to SDPose that can be passed directly to the main script:

`--device`: device to run inference on. Accepts `cpu` or `gpu`. Defaults to `cpu` if not specified.

`--num-workers N`: number of parallel workers for processing videos (default: 1).

## Model and code details

The SDPose model is used, producing **133 keypoints** in the COCO-Wholebody-133 format. 

Each keypoint has 3 coordinates (x, y, z) in the original SDPose implementation, 

Pose estimation is performed via the `videos_to_poses` command from a branch of the 
[`pose-format`] library located [`here`](https://github.com/catherine-o-brien/pose/tree/new_estimators). The code is based upon SDPose's [Gradio implementation](https://huggingface.co/spaces/teemosliang/SDPose). Notably, while the original implementation of SDPose can detect multiple people, ours accepts only the detected person with the highest keypoint confidence.

Pose estimation is performed via the `videos_to_poses` command from a fork of the
[`pose-format`](https://github.com/catherine-o-brien/pose) library (`new_estimators` branch),
using `--format sdpose`.

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA drivers (optional, if `--device gpu` is used)

## Cite

```bibtex
@inproceedings{li2024sdpose,
  title={SDPose: Tokenized Pose Estimation via Circulation-Guide Self-Distillation},
  author={Li, Sichen and He, Yuer and Liang, Jiajun and Zhang, Feixiang and Liu, Shaohua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
