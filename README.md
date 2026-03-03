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
bash install.sh --type openpose [--slurm]
```

Then to estimate poses for a folder of input videos:

```bash
bash videos_to_poses.sh \
    --type openpose \
    --input /path/to/videos \
    --output /path/to/poses [--extra] [--slurm]
```

### Arguments

| Parameter |   |
|----------|---|
| --type | Which pose estimator to use. Available choices: `openpose`, `mediapipe`  |
| --input  | Path to folder of videos  |
| --output | Path to folder of .pose files  |
| --extra (optional)   |  Additional arguments passed on to the estimator-specific run script |
| --slurm (optional) | Whether to submit estimation jobs to a SLURM queue instead of executing directly  |

Installing may also differ if `--slurm` is used, so if you are working on a SLURM cluster, also
use `--slurm` for the installation script above.

For additional, estimator-specific arguments that can be passed via `--extra`, see the respective estimator's
README file (in the `estimators` sub-folders).

For instance, for the `openpose` estimator, if `--slurm` is used, a further argument `--chunks` can be passed like so:

```bash
bash videos_to_poses.sh \
    --type openpose \
    --input /path/to/videos \
    --output /path/to/poses \
    --slurm
    --extra --chunks 20
```

to specify the number of chunks to split the input videos into, and the number of resulting jobs to submit.

## Pose estimators included

| Estimator  | Exact version, details  | Requirements |
|------------|-------------------------|--------------|
| openpose   | 137-keypoint model | Singularity CE >= 3.7 or Apptainer, NVIDIA GPU with driver supporting CUDA 11.x |
| mediapipe  | MediaPipe Holistic, model complexity 2, with iris refinement | Python 3 |

## Details on output pose format

Estimated poses are converted to the binary `.pose` format of the [`pose-format` library](https://github.com/sign-language-processing/pose). All `pose-format`
utilities to store, load, manipulate and visualize the poses can be applied.

## SLURM usage

If `--slurm` is used, then pose estimation jobs are submitted with SLURM commands, assuming a SLURM login node that has `sbatch`, for instance.
These commands are meant as an example and are tailored to a UZH SLURM cluster. They are not expected to run without some modifications on
other clusters.

## Testing

Estimator-specific tests live in the `test/` directory. See [`test/README.md`](test/README.md) for details on how to run them.

## Acknowledgements

(cite paper once on Arxiv)

```bibtex
@misc{obrien-et-al-2026video-to-pose, 
    title={Convenience code for installing and using several pose estimation systems},
    author={O'Brien, Catherine and Sant, Gerard and M{\"u}ller, Mathias},
    howpublished={\url{https://github.com/ZurichNLP/video-to-pose}},
    year={2026}
}
```
