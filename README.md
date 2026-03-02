# video-to-pose tools

This library simplifies installing and using several pose estimation systems.

## Download

```bash
git clone https://github.com/ZurichNLP/video-to-pose
cd video-to-pose
```

## How to use

To install a specific estimator on your system:

```bash
install.sh --type openpose [--slurm]
```

Then to estimate poses for a folder of input videos:

```bash
videos_to_poses.sh --type openpose --input /path/to/videos --output /path/to/poses [--args] [--slurm]
```

### Arguments

| Parameter |   |
|----------|---|
| --type | Which pose estimator to use. Available choices: `openpose`, ...  |
| --input  | Path to folder of videos  |
| --output | Path to folder of .pose files  |
| --args (optional)   |  Additional arguments passed on to the estimator-specific run script |
| --slurm (optional) | Whether to submit estimation jobs to a SLURM queue instead of executing directly  |

## Pose estimators included

| Estimator  | Exact version, details  |
|----------|---|
| openpose |  137-keypoint model |

## Details on output pose format

Estimated poses are converted to the binary `.pose` format of the [`pose-format` library](https://github.com/sign-language-processing/pose). All `pose-format`
utilities to store, load, manipulate and visualize the poses can be applied.

## SLURM usage

If `--slurm` is used, then pose estimation jobs are submitted with SLURM commands, assuming a SLURM login node that has `sbatch`, for instance.
These commands are meant as an example and are tailored to a UZH SLURM cluster. They are not expected to run without some modifications on
other clusters.

## Acknowledgements

(cite paper once on Arxiv)
