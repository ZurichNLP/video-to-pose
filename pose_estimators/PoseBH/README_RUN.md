# PoseBH Run Guide (Video Batch Inference)

This guide focuses on running PoseBH for large video datasets and exporting COCO-WholeBody JSON results.

## Main scripts

- `video_inference_wholebody_mp.py`: recommended multi-process batch inference script.
- `posebh_popsign.sh`: SLURM template for PopSign.
- `posebh_citizen.sh`: SLURM template for ASL Citizen.
- `posebh_semlex.sh`: SLURM template for Sem-Lex.
- `split.sh`: helper script to split multi-dataset checkpoint (`tools/model_split.py`).

## 1. Environment setup

From `pose_estimators/PoseBH`:

```bash
conda create -n posebh python=3.8 -y
conda activate posebh

# PyTorch (example: CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# MMCV version used by this repo
pip install "git+https://github.com/open-mmlab/mmcv.git@v1.4.8"

# Install PoseBH/MMPose package in editable mode
pip install -v -e .

# Project dependencies
pip install -r requirements.txt

# Needed when using --use_yolo in video_inference_wholebody_mp.py
pip install ultralytics
```

Notes:

- `video_inference_wholebody_mp.py` can use YOLO (`--use_yolo`) or mmdet detector.
- If YOLO weights are missing, Ultralytics will download them automatically.

## 2. Checkpoint placement

Expected defaults:

- Pose model config:
  - `configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py`
- Pose checkpoint:
  - `weights/posebh/wholebody.pth`
- YOLO checkpoint (when `--use_yolo`):
  - `yolov8n.pt`

## 3. Run locally

From `pose_estimators/PoseBH`:

```bash
python video_inference_wholebody_mp.py \
  --input_dir /path/to/videos \
  --output_dir /path/to/output \
  --pose_config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
  --pose_checkpoint weights/posebh/wholebody.pth \
  --use_yolo \
  --yolo_checkpoint yolov8n.pt \
  --device cuda:0 \
  --num_workers 3 \
  --batch_size 4 \
  --queue_size 8 \
  --frame_interval 1 \
  --resume
```

## 4. Run on SLURM

Use the provided templates:

```bash
sbatch posebh_popsign.sh
sbatch posebh_citizen.sh
sbatch posebh_semlex.sh
```

Adjust:

- `module load ...`
- conda env name (`posebh`)
- `--input_dir` and `--output_dir`
- resource settings (`cpus`, `gpus`, `mem`, time)

## 5. Output behavior

- One JSON file per video in COCO-WholeBody style.
- Output filename suffix: `_wholebody.json`.
- Relative folder structure from input is preserved.
- `--resume` skips videos that already have output JSON.

## 6. Performance tuning

Most useful flags:

- `--num_workers`: CPU decode workers.
- `--batch_size`: number of frames per GPU inference batch.
- `--queue_size`: frame queue size between reader and GPU worker.
- `--frame_interval`: process every N-th frame.

If OOM or instability occurs:

- reduce `--batch_size` (first choice)
- reduce `--queue_size`
- reduce `--num_workers`

## 7. Optional: split multi-dataset checkpoint

To split a multi-dataset checkpoint into per-dataset heads:

```bash
python tools/model_split.py --source /path/to/your_checkpoint.pth
```

Or use the included SLURM helper:

```bash
sbatch split.sh
```

## 8. Troubleshooting

- `Input directory does not exist`: fix `--input_dir`.
- `No module named mmpose/mmdet`: ensure editable install and dependencies are installed in the active conda env.
- `CUDA not available`: check GPU allocation and CUDA-compatible PyTorch build.
- detector download failures (offline nodes): pre-download required checkpoints (`wholebody.pth`, `yolov8n.pt`) to local paths.
