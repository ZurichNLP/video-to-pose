from huggingface_hub import hf_hub_download
import shutil
import os

target_dir = "/home/gsantm/store/pose_estimators/SMPLest-X/pretrained_models"
os.makedirs(target_dir, exist_ok=True)

tmp_path = hf_hub_download(
    repo_id="waanqii/SMPLest-X",
    filename="smplest_x_h.pth.tar"
)

shutil.copy(tmp_path, target_dir)