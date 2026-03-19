# OpenPose

## Estimator-specific arguments

Additional arguments specific to OpenPose that can be passed directly to the main script:

`--chunks`: to specify the number of chunks to split the input videos into, and the number of resulting
jobs to submit. Only allowed together with `--slurm`.

## Model and code details

The 137-keypoint OpenPose model will  be used (as opposed to the 135-keypoint model).

Caveat: sometimes OpenPose will detect several people even in frames where only one person is visible.
Downstream processing should expect this edge case.

The scripts in this directory re-use code from https://github.com/bricksdont/openpose-singularity-uzh/, which
uses a pre-built docker image for OpenPose, based on the original OpenPose source code
(https://github.com/CMU-Perceptual-Computing-Lab/openpose).

## Requirements

- Singularity CE >= 3.x or Apptainer >= 1.x
- NVIDIA GPU with CUDA drivers

OpenPose requires a GPU to run. Making it run on CPU is not possible with reasonable effort in our
experience, and will be very slow.

## Cite

```bibtex
@article{cao2019openpose,
  title={Openpose: Realtime multi-person 2d pose estimation using part affinity fields},
  author={Cao, Zhe and Hidalgo, Gines and Simon, Tomas and Wei, Shih-En and Sheikh, Yaser},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={43},
  number={1},
  pages={172--186},
  year={2019},
  publisher={IEEE}
}
```

```bibtex
@misc{muller2026-openpose-singularity-uzh, 
    title={OpenPose with Singularity or Apptainer (UZH)},
    author={M{\"u}ller, Mathias},
    howpublished={\url{https://github.com/bricksdont/openpose-singularity-uzh}},
    year={2026}
}
```