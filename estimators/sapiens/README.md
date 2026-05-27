# Sapiens

## Estimator-specific arguments

Additional arguments specific to Sapiens that can be passed directly to the main script:

`--device`: device to run inference on. Accepts `cpu` or `gpu`. Defaults to `gpu` if a CUDA device is available, otherwise falls back to CPU. **Please note that CPU inference on Sapiens is extremely slow.**

`--num-workers N`: number of parallel workers for processing videos (default: 1). Only useful with `--device cpu`; with GPU, multiple workers compete for GPU memory and will likely cause out-of-memory errors.

## Model and code details

This runs pose estimation using Meta's [Sapiens](https://github.com/facebookresearch/sapiens) `1B` pose model, trained on the **Goliath** keypoint set.

Sapiens-Goliath defines 308 keypoints. In this repo, the two wrist points are duplicated so that each wrist appears in both the body and the corresponding hand component for compatibility with the `.pose` format, giving **310 keypoints** total.

| Component   | Keypoints |
|-------------|-----------|
| Body (incl. wrists) | 23  |
| Left hand   | 21        |
| Right hand  | 21        |
| Body extra  | 7         |
| Face        | 150       |
| Left ear    | 26        |
| Right ear   | 26        |
| Left iris   | 9         |
| Right iris  | 9         |
| Left pupil  | 9         |
| Right pupil | 9         |
| **Total**   | **310**   |

Output shape: `(frames, people, 310, 2)` with a separate `confidence` array of shape `(frames, people, 310)`. These outputs are converted into the `.pose` format. The `people` dimension is always 1. If two people are detected, this implementation will only accept the one with the higher confidence value. 

Pose estimation is performed via the `videos_to_poses` command from a fork of the
[`pose-format`](https://github.com/sign-language-processing/pose) library located [here](https://github.com/catherine-o-brien/pose/tree/new_estimators), which wraps the Sapiens model.

The model weights (`sapiens-pose-1b`, TorchScript, several GB) are hosted on Hugging Face at [`facebook/sapiens-pose-1b-torchscript`](https://huggingface.co/facebook/sapiens-pose-1b-torchscript) and downloaded on first run. Because upstream renamed the checkpoint file on the HF `main` branch, `install_sapiens.sh` pre-downloads a pinned snapshot of the model during installation to avoid a download failure at run time.

## Requirements
- NVIDIA GPU with CUDA drivers (not required if `--device cpu` is used, but strongly recommended — the 1B model is large and slow on CPU)
- Enough disk space and bandwidth to download the multi-GB model checkpoint

### Cluster-specific notes (if using the flag `--slurm`)

When `--slurm` is passed, `install_sapiens.sh` loads the `miniforge3` module and, if a GPU is detected, `module load cuda/12.6.3`. If this is incorrect syntax on your cluster, load CUDA manually before running the install script. Outside of a SLURM environment, no `module load` commands are run.

## Cite
```bibtex
@inproceedings{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Su, Zhaoen and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
