# ViTPose (Batch Video to COCO-WholeBody JSON)

This folder contains batch inference scripts that run ViTPose + YOLO detection on videos and export **COCO-WholeBody (133 keypoints)** JSON files.

## Files in this folder

- `vitpose_batch_processor.py`: PopSign pipeline (uses streaming mode to reduce memory usage).
- `citizen_vitpose_batch_processor.py`: ASL Citizen pipeline.
- `semlex_vitpose_batch_processor.py`: Sem-Lex pipeline.
- `vitpose_popsign.sh`, `citizen_vitpose_popsign.sh`, `semlex_vitpose_popsign.sh`: SLURM job wrappers.
- `ckpts/`: model checkpoints (`vitpose-h-wholebody.pth`, `yolov8n.pt`).

## 1. Create environment

From `pose_estimators/vitpose`:

```bash
conda create -n vitpose python=3.10 -y
conda activate vitpose

# Install PyTorch for your CUDA/CPU setup first (example: CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Project requirements
pip install -r requirements.txt
pip install -r easy_ViTPose/requirements.txt
pip install -e easy_ViTPose
```

If you run on CPU only, install CPU PyTorch wheels instead of CUDA wheels.

## 2. Configure input/output directories

Each script has hard-coded paths at the top:

- `INPUT_VIDEO_PATH`
- `OUTPUT_JSON_PATH`

Edit them before running. Example locations:

- `vitpose_batch_processor.py` (PopSign)
- `citizen_vitpose_batch_processor.py` (ASL Citizen)
- `semlex_vitpose_batch_processor.py` (Sem-Lex)

## 3. Run locally

From `pose_estimators/vitpose`:

```bash
python vitpose_batch_processor.py
```

Or:

```bash
python citizen_vitpose_batch_processor.py
python semlex_vitpose_batch_processor.py
```

## 4. Run with SLURM

The provided job scripts expect a conda env named `vitpose`:

```bash
sbatch vitpose_popsign.sh
sbatch citizen_vitpose_popsign.sh
sbatch semlex_vitpose_popsign.sh
```

Update module names/resources in these `.sh` files if your cluster setup is different.

## 5. Output behavior

- Output JSON keeps the same relative folder structure as the input videos.
- Existing JSON files are skipped automatically (resume-friendly).
- Model files are auto-downloaded if missing:
  - ViTPose: `ckpts/vitpose-h-wholebody.pth`
  - YOLO: `ckpts/yolov8n.pt`

## 6. Notes

- Supported video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`
- The scripts prefer `decord` and fall back to OpenCV when needed.
- `vitpose_batch_processor.py` is the safer choice for large datasets because it uses streaming mode and lower memory pressure.

## 7. Quick troubleshooting

- `Input path not found`: update `INPUT_VIDEO_PATH`.
- `easy_ViTPose not available`: make sure `pip install -e easy_ViTPose` was run in this folder.
- CUDA not used: check PyTorch install and `torch.cuda.is_available()`.
- Download blocked/offline cluster: place required checkpoints manually in `ckpts/`.
