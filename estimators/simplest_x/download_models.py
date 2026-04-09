#!/usr/bin/env python3
"""Download SMPLest-X pretrained models and SMPL-X body models via HuggingFace Hub."""
import os
import sys
import shutil
from pathlib import Path


def link_if_missing(src, dst):
    """Symlink src -> dst if dst doesn't exist. Avoids duplicating large files on disk."""
    dst = Path(dst)
    if not dst.exists():
        os.symlink(src, dst)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <smplest_x_repo_root>", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import hf_hub_download

    repo_root = Path(sys.argv[1]).resolve()
    pretrain_dir = repo_root / "pretrained_models"
    smpl_dir = pretrain_dir / "smplest_x_h"
    human_dir = repo_root / "human_models" / "smplx"

    pretrain_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoint (~8.2 GB) — symlink to avoid duplicating on disk
    print("Downloading SMPLest-X checkpoint (~8.2 GB)...")
    ckpt = hf_hub_download(repo_id="waanqii/SMPLest-X", filename="smplest_x_h.pth.tar")
    link_if_missing(ckpt, smpl_dir / "smplest_x_h.pth.tar")

    # Download config — real copy because we patch it
    print("Downloading config...")
    config_src = hf_hub_download(repo_id="waanqii/SMPLest-X", filename="config_base.py")
    target_config = smpl_dir / "config_base.py"
    shutil.copy(config_src, target_config)

    text = target_config.read_text()
    absolute_human_path = str(repo_root / "human_models")
    text = text.replace(
        "'./human_models/human_model_files'",
        f"'{absolute_human_path}'",
    )
    target_config.write_text(text)

    # Download YOLOv8x detector weights — symlink
    print("Downloading YOLOv8x detector...")
    yolo = hf_hub_download(repo_id="Ultralytics/YOLOv8", filename="yolov8x.pt")
    link_if_missing(yolo, pretrain_dir / "yolov8x.pt")

    # Download SMPL-X body models — symlink
    print("Downloading SMPL-X body models...")
    smplx_files = [
        "SMPLX_FEMALE.npz",
        "SMPLX_FEMALE.pkl",
        "SMPLX_MALE.npz",
        "SMPLX_MALE.pkl",
        "SMPLX_NEUTRAL.npz",
        "SMPLX_NEUTRAL.pkl",
    ]
    for fname in smplx_files:
        src = hf_hub_download(repo_id="lilpotat/pytorch3d", filename=f"models/{fname}")
        link_if_missing(src, human_dir / fname)

    print("All pretrained assets ready.")


if __name__ == "__main__":
    main()
