---
title: SDPose
emoji: 🚀
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
hf_oauth: true
license: mit
short_description: Implementation demo of SDPose-OOD.
---

# SDPose: How to Run (Batch Inference + Optional Gradio Demo)

This folder includes two main usage modes:

- Batch dataset processing (recommended for large video collections)
- Interactive Gradio demo (`app.py`)

## What each script does

- `conbatch.py`: memory-optimized streaming pipeline (best default for large datasets).
- `conbatch_citizen.py`: prefetch-based batch pipeline for ASL Citizen paths.
- `conbatch_semlex.py`: prefetch-based batch pipeline for Sem-Lex paths.
- `trainpopsign.sh`, `traincitizen.sh`, `trainsemlex.sh`: SLURM wrappers for the three batch scripts.

All batch scripts:

- recursively scan the input directory for videos
- skip already processed videos (resume-safe)
- export one COCO-style `.json` per video

## 1. Create and activate environment

From `pose_estimators/SDPose`:

```bash
conda create -n SDPose python=3.10 -y
conda activate SDPose
pip install -r requirements.txt
```

Optional:

```bash
pip install xformers==0.0.25 --no-build-isolation
```

## 2. Prepare paths (required)

Before running, edit the constants at the top of your target script:

- `REMOTE_BASE_PATH`: input video root directory
- `LOCAL_OUTPUT_PATH`: output directory for JSON files

Default mappings:

- `conbatch.py` -> PopSign
- `conbatch_citizen.py` -> ASL Citizen
- `conbatch_semlex.py` -> Sem-Lex

## 3. Run locally

From `pose_estimators/SDPose`:

```bash
python conbatch.py
```

Or dataset-specific variants:

```bash
python conbatch_citizen.py
python conbatch_semlex.py
```

## 4. Run on SLURM

Use the provided scripts:

```bash
sbatch trainpopsign.sh
sbatch traincitizen.sh
sbatch trainsemlex.sh
```

These job files expect:

- module `miniforge3`
- conda environment name `SDPose`

Adjust module names/resources if your cluster uses different settings.

## 5. Model download behavior

At first run, scripts download:

- SDPose weights from Hugging Face repo `teemosliang/SDPose-Wholebody` into `./model_cache`
- YOLO weights (`yolo11x.pt`) if not found locally

If your compute node is offline, pre-populate `model_cache/` and the YOLO weight file in this directory.

## 6. Output format

Each processed video produces one JSON file with:

- `images`
- `annotations`
- `categories`

Relative folder structure from input is preserved in output.

## 7. Performance tuning

Important knobs in scripts:

- `BATCH_SIZE`
- `CHUNK_SIZE` (only in `conbatch.py`)
- `NUM_CPU_WORKERS` (prefetch pipelines)
- `ENABLE_FP16`

Quick guidance:

- GPU OOM: reduce `BATCH_SIZE`; for `conbatch.py`, also reduce `CHUNK_SIZE`.
- High CPU RAM usage: prefer `conbatch.py` (streaming mode).

## 8. Optional: run the Gradio demo

```bash
python app.py
```

Then open the local Gradio URL shown in terminal.

## 9. Troubleshooting

- `Path not found`: verify `REMOTE_BASE_PATH` and `LOCAL_OUTPUT_PATH`.
- `CUDA not available`: check GPU allocation and CUDA-enabled PyTorch install.
- `YOLO failed`: ensure `ultralytics` is installed and model download is allowed.
- `snapshot_download` errors: check Hugging Face connectivity/permissions on your cluster.
