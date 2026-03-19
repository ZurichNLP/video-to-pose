# AlphaPose

## Estimator-specific arguments

Additional arguments specific to AlphaPose that can be passed directly to the main script:

`--chunks`: number of parallel SLURM jobs to split the input videos into. Only allowed together with `--slurm`.

`--keypoints`: keypoint format to use. Choices: `136` (HALPE_136, default) or `133` (COCO WholeBody).

`--lowprio`: submit SLURM jobs to a low-priority partition. Only allowed together with `--slurm`.

## Model and code details

The default model is the 136-keypoint HALPE_136 whole-body model (Multi-domain DCN Combined, trained on HALPE and COCO WholeBody), covering body, face, hands, and feet. A 133-keypoint COCO WholeBody variant can be selected via `--keypoints 133`.

The scripts in this directory re-use code from https://github.com/bricksdont/alphapose-singularity-uzh, which packages AlphaPose in a Singularity/Apptainer container based on the original AlphaPose source code (https://github.com/MVIG-SJTU/AlphaPose).

## Requirements

- Singularity CE >= 3.x or Apptainer >= 1.x
- NVIDIA GPU with CUDA drivers (CPU mode is not supported — AlphaPose uses CUDA-only Deformable Convolutions)
- ~8 GB disk space for the container image (pulled from GHCR) plus additional space for model weights.
Building the container from scratch (usually not required) requires ~35 GB.

## Cite

```bibtex
@article{fang2022alphapose,
  title={Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time},
  author={Fang, Hao-Shu and Li, Jiefeng and Tang, Hang and Xu, Chao and Zhu, Haoyi and Xiu, Yunlian and Li, Yanfeng and Lu, Cewu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={6},
  pages={7157--7173},
  year={2022},
  publisher={IEEE}
}
```

```bibtex
@misc{muller-et-al-2026alphapose-singularity-uzh,
    title={Singularity/Apptainer container pipeline for running AlphaPose whole-body pose estimation},
    author={M{\"u}ller, Mathias and Sant, Gerard},
    howpublished={\url{https://github.com/bricksdont/alphapose-singularity-uzh}},
    year={2026}
}
```