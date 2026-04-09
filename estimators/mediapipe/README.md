# Mediapipe Holistic

## Estimator-specific arguments

Additional arguments specific to MediaPipe that can be passed directly to the main script:

`--num-workers`: number of parallel workers for processing videos. Defaults to 1.

## Model and code details

The MediaPipe Holistic model is used, producing **586 keypoints** in total across 5 components:

| Component           | Keypoints |
|---------------------|-----------|
| Pose landmarks      | 33        |
| Face landmarks      | 478       |
| Left hand landmarks | 21        |
| Right hand landmarks| 21        |
| Pose world landmarks| 33        |
| **Total**           | **586**   |

Each keypoint has 3 coordinates (x, y, z). Only 1 person is detected per frame.

Pose estimation is performed via the `videos_to_poses` command from the 
[`pose-format`](https://github.com/sign-language-processing/pose) library, which is a wrapper around the MediaPipe model.


The model is run with `model_complexity=2`, `smooth_landmarks=false`, and `refine_face_landmarks=true` (this
behaviour is not configurable). If you need more fine-grained control over this, use the original
code from [`pose-format`](https://github.com/sign-language-processing/pose).


## Requirements

- Python 3
- No GPU required. 

## Cite

```bibtex
@misc{lugaresi2019mediapipe,
  title={MediaPipe: A Framework for Building Perception Pipelines},
  author={Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and McClanahan, Chris and Uboweja, Esha and Hays, Michael and Zhang, Fan and Chang, Chuo-Ling and Yong, Ming Guang and Lee, Juhyun and Chang, Wan-Teh and Hua, Wei and Georg, Manfred and Grundmann, Matthias},
  year={2019},
  eprint={1906.08172},
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```

```bibtex
@misc{moryossef2021pose-format, 
    title={pose-format: Library for viewing, augmenting, and handling .pose files},
    author={Moryossef, Amit and M\"{u}ller, Mathias and Fahrni, Rebecka},
    howpublished={\url{https://github.com/sign-language-processing/pose}},
    year={2021}
}
```