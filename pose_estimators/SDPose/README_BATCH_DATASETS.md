# SDPose Batch Dataset Processing

This document explains how to run SDPose in batch mode for large video datasets.

## Available Batch Pipelines

- `conbatch.py`: memory-optimized streaming pipeline (recommended for large datasets and lower CPU RAM).
- `conbatch_citizen.py`: prefetch-based batch pipeline for the Citizen dataset path.
- `conbatch_semlex.py`: prefetch-based batch pipeline for the Sem-Lex dataset path.

Each script scans a dataset directory, skips already processed videos, and writes one COCO-style `.json` per video.

## 1) Environment Setup

From `pose_estimators/SDPose`:

```bash
conda create -n SDPose python=3.10 -y
conda activate SDPose
pip install -r requirements.txt
pip install decord
```

Optional performance package:

```bash
pip install xformers==0.0.25 --no-build-isolation
```

## 2) Configure Paths and Runtime Parameters

Open the target script and edit these constants:

- `REMOTE_BASE_PATH`: input dataset root (video files)
- `LOCAL_OUTPUT_PATH`: output root for generated `.json`
- `VIDEO_EXTENSIONS`: accepted video suffixes

Key performance parameters:

- `BATCH_SIZE`
- `CHUNK_SIZE` (only in `conbatch.py`)
- `NUM_CPU_WORKERS` (in prefetch pipelines)
- `ENABLE_FP16`

## 3) Run Locally

```bash
cd pose_estimators/SDPose
python conbatch.py
```

Alternative datasets:

```bash
python conbatch_citizen.py
python conbatch_semlex.py
```

## 4) Run on SLURM

Preconfigured jobs:

```bash
sbatch train.sh
sbatch traincitizen.sh
sbatch trainsemlex.sh
```

These scripts activate the `SDPose` conda environment and run the corresponding batch script.

## 5) Output Format

For each input video, the pipeline writes one COCO-style JSON file containing:

- `images`
- `annotations`
- `categories`

Output file naming mirrors relative input paths, with `.json` extension.

## 6) Resume Behavior

All batch scripts call `check_existing(...)` and skip videos that already have output JSON.
This means interrupted runs can be resumed safely without reprocessing completed videos.

## 7) Troubleshooting

- `CUDA not available`: verify GPU node selection and CUDA-enabled PyTorch.
- `Decord not available`: install with `pip install decord` (pipeline still runs with cv2 fallback in some cases).
- GPU OOM: lower `BATCH_SIZE`; for `conbatch.py`, also lower `CHUNK_SIZE`.
- Slow processing: increase `NUM_CPU_WORKERS` carefully and monitor RAM usage.
