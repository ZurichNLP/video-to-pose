#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViTPose Batch Video Processor for COCO-WholeBody 133 Keypoints
==============================================================
- Automatically download ViTPose and YOLO models
- Prefer Decord for video decoding, and fall back to OpenCV on failure
- Support batch processing for video directories
- Export JSON in COCO-WholeBody 133 keypoint format
- Use a prefetch queue to parallelize reading and inference
"""

import os
import sys

# ============================================================
# Path setup: ensure easy_ViTPose can be imported correctly
# ============================================================
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# easy_ViTPose repository path (relative to script location)
# If the script is under ~/pose2cla/vitpose/, the easy_ViTPose package is under ~/pose2cla/vitpose/easy_ViTPose/
EASY_VITPOSE_REPO = os.path.join(SCRIPT_DIR, "easy_ViTPose")

# Add the easy_ViTPose repository directory to the Python path
if os.path.isdir(EASY_VITPOSE_REPO):
    sys.path.insert(0, EASY_VITPOSE_REPO)
    print(f"📁 Added to path: {EASY_VITPOSE_REPO}")
import cv2
import numpy as np
import torch
import json
import time
import queue
import threading
import gc
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not available, install with: pip install tqdm")

# Decord
try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("native")
    DECORD_AVAILABLE = True
    print("✅ Decord available")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Decord not available, using OpenCV for all videos")

# easy_ViTPose - import fix
VitInference = None
VITPOSE_AVAILABLE = False

try:
    # Method 1: try importing directly from the package (if __init__.py exports correctly)
    from easy_ViTPose import VitInference
    VITPOSE_AVAILABLE = True
    print("✅ easy_ViTPose available (from package)")
except ImportError:
    pass

if not VITPOSE_AVAILABLE:
    try:
        # Method 2: import from the inference module (full path)
        from easy_ViTPose.inference import VitInference
        VITPOSE_AVAILABLE = True
        print("✅ easy_ViTPose available (from easy_ViTPose.inference)")
    except ImportError:
        pass

if not VITPOSE_AVAILABLE:
    try:
        # Method 3: try inference under vit_utils
        from easy_ViTPose.vit_utils.inference import VitInference
        VITPOSE_AVAILABLE = True
        print("✅ easy_ViTPose available (from vit_utils.inference)")
    except ImportError:
        pass

if not VITPOSE_AVAILABLE:
    print("❌ easy_ViTPose not available.")
    print("   Please install correctly:")
    print("   git clone https://github.com/JunkyByte/easy_ViTPose.git")
    print("   cd easy_ViTPose && pip install -e .")
    print("   pip install -r requirements.txt")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================
# Input/output paths
INPUT_VIDEO_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/"
OUTPUT_JSON_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/ViTPose/"

# Model configuration
MODEL_DIR = "./ckpts"
VITPOSE_MODEL_NAME = "vitpose-h-wholebody.pth"
VITPOSE_MODEL_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/wholebody/vitpose-h-wholebody.pth"
VITPOSE_MODEL_SIZE = "h"  # 's', 'b', 'l', 'h'
VITPOSE_DATASET = "wholebody"  # 133 keypoints

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# Video formats
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

# Performance configuration
NUM_CPU_WORKERS = 4
YOLO_SIZE = 320
DECORD_NUM_THREADS = 4
PREFETCH_QUEUE_SIZE = 1  # Reduce prefetch queue size to lower memory usage

# Memory optimization configuration
ENABLE_PREFETCH = False  # Set to False to disable prefetch and reduce memory usage
MAX_FRAMES_IN_MEMORY = 500  # Videos with more than this many frames will be processed frame by frame
CLEAR_MEMORY_INTERVAL = 5  # Clear memory every N processed videos

# Format categories
DECORD_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
CV2_FORMATS = {'.webm', '.flv', '.wmv'}


# ============================================================
# COCO-WholeBody 133 keypoint definitions
# ============================================================
COCO_WHOLEBODY_KEYPOINTS = [
    # Body (17)
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    # Feet (6)
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    # Face (68)
    *[f"face_{i}" for i in range(68)],
    # Left hand (21)
    *[f"left_hand_{i}" for i in range(21)],
    # Right hand (21)
    *[f"right_hand_{i}" for i in range(21)],
]

COCO_WHOLEBODY_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
    [16, 18], [16, 19], [16, 20], [17, 21], [17, 22], [17, 23],
]


# ============================================================
# Model download utilities
# ============================================================
def download_file(url: str, save_path: str, desc: str = None) -> bool:
    """Download a file with progress bar support"""
    import urllib.request
    import shutil

    if desc is None:
        desc = os.path.basename(save_path)

    print(f"📥 Downloading {desc}...")
    print(f"   URL: {url}")
    print(f"   Save to: {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = save_path + ".tmp"

    try:
        if TQDM_AVAILABLE:
            with urllib.request.urlopen(url) as response:
                file_size = int(response.headers.get('Content-Length', 0))

            with urllib.request.urlopen(url) as response:
                with open(temp_path, 'wb') as out_file:
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))
        else:
            print("   (Install tqdm for progress bar: pip install tqdm)")
            urllib.request.urlretrieve(url, temp_path)

        shutil.move(temp_path, save_path)
        print(f"✅ Downloaded successfully: {save_path}")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def download_with_huggingface_hub(url: str, save_path: str) -> bool:
    """Download via huggingface_hub"""
    try:
        from huggingface_hub import hf_hub_download

        if "huggingface.co" in url:
            parts = url.split("huggingface.co/")[1].split("/resolve/main/")
            repo_id = parts[0]
            filename = parts[1]

            print(f"📥 Downloading from Hugging Face Hub...")
            print(f"   Repo: {repo_id}")
            print(f"   File: {filename}")

            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(save_path),
                local_dir_use_symlinks=False
            )

            if os.path.abspath(cached_path) != os.path.abspath(save_path):
                import shutil
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(cached_path, save_path)

            print(f"✅ Downloaded successfully: {save_path}")
            return True

    except ImportError:
        print("   huggingface_hub not available, using urllib...")
        return False
    except Exception as e:
        print(f"   huggingface_hub download failed: {e}, trying urllib...")
        return False

    return False


def ensure_model_exists(model_path: str, model_url: str, model_name: str) -> bool:
    """Ensure model files exist, download if missing"""
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ {model_name} found: {model_path} ({file_size:.1f} MB)")
        return True

    print(f"⚠️ {model_name} not found: {model_path}")
    print(f"   Will download from: {model_url}")

    if "huggingface.co" in model_url:
        if download_with_huggingface_hub(model_url, model_path):
            return True

    return download_file(model_url, model_path, model_name)


def setup_models() -> Tuple[Optional[str], Optional[str]]:
    """Set and return model paths, downloading models when necessary"""
    print("\n" + "=" * 50)
    print("🔧 Setting up models")
    print("=" * 50)

    os.makedirs(MODEL_DIR, exist_ok=True)

    vitpose_model_path = os.path.join(MODEL_DIR, VITPOSE_MODEL_NAME)
    yolo_model_path = os.path.join(MODEL_DIR, YOLO_MODEL_NAME)

    print("\n📦 Checking ViTPose model...")
    if not ensure_model_exists(vitpose_model_path, VITPOSE_MODEL_URL, "ViTPose"):
        print("❌ Failed to get ViTPose model")
        return None, None

    print("\n📦 Checking YOLO model...")
    if not ensure_model_exists(yolo_model_path, YOLO_MODEL_URL, "YOLO"):
        print("❌ Failed to get YOLO model")
        return None, None

    print("\n✅ All models ready!")
    return vitpose_model_path, yolo_model_path


# ============================================================
# Video readers
# ============================================================
class VideoReaderFactory:
    """Create the optimal reader based on video format"""

    @staticmethod
    def create(video_path: str):
        ext = os.path.splitext(video_path)[1].lower()

        if ext in DECORD_FORMATS and DECORD_AVAILABLE:
            return DecordVideoReader(video_path)
        else:
            return CV2VideoReader(video_path)


class DecordVideoReader:
    """Decord video reader - suitable for mp4/avi/mov/mkv"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.vr = None
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.ready = False
        self._load()

    def _load(self):
        try:
            ctx = cpu(0)
            self.vr = VideoReader(self.video_path, ctx=ctx, num_threads=DECORD_NUM_THREADS)
            self.total_frames = len(self.vr)

            if self.total_frames > 0:
                first = self.vr[0]
                self.height, self.width = first.shape[:2]
                self.fps = self.vr.get_avg_fps() or 30.0
                self.ready = True
        except Exception as e:
            print(f"⚠️ Decord failed for {self.video_path}: {e}, falling back to OpenCV")
            self._fallback_cv2()

    def _fallback_cv2(self):
        """Use cv2 to read metadata when Decord fails"""
        self.vr = None
        self.ready = False

        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.total_frames <= 0:
                self.total_frames = 0
                while cap.read()[0]:
                    self.total_frames += 1
        cap.release()

    def read_all(self) -> List[np.ndarray]:
        """Read all frames at once and return RGB format"""
        if self.total_frames <= 0:
            return []

        if self.ready and self.vr is not None:
            try:
                indices = list(range(self.total_frames))
                frames_rgb = self.vr.get_batch(indices)

                if hasattr(frames_rgb, 'asnumpy'):
                    frames_rgb = frames_rgb.asnumpy()

                return [frames_rgb[i] for i in range(len(frames_rgb))]
            except Exception as e:
                print(f"⚠️ Decord batch read failed: {e}")

        return self._read_all_cv2()

    def _read_all_cv2(self) -> List[np.ndarray]:
        """Read all frames using OpenCV"""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def close(self):
        self.vr = None


class CV2VideoReader:
    """OpenCV video reader - suitable for webm and similar formats"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self._get_info()

    def _get_info(self):
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                frame_count = 0
                while cap.read()[0]:
                    frame_count += 1
            self.total_frames = frame_count
        cap.release()

    def read_all(self) -> List[np.ndarray]:
        """Read all frames and return RGB format"""
        if self.total_frames <= 0:
            return []

        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def close(self):
        pass


# ============================================================
# Prefetch processor
# ============================================================
class PrefetchVideoProcessor:
    """Video processor with prefetching"""

    def __init__(self, video_list: List[Tuple[str, str]]):
        self.video_list = video_list
        self.prefetch_queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
        self.stop_flag = threading.Event()
        self.prefetch_thread = None

    def _prefetch_worker(self):
        """Background prefetch thread"""
        for video_path, output_path in self.video_list:
            if self.stop_flag.is_set():
                break

            try:
                reader = VideoReaderFactory.create(video_path)
                frames = reader.read_all()
                info = {
                    'video_path': video_path,
                    'output_path': output_path,
                    'frames': frames,
                    'width': reader.width,
                    'height': reader.height,
                    'fps': reader.fps,
                    'total_frames': len(frames)
                }
                reader.close()
                self.prefetch_queue.put(info)

            except Exception as e:
                print(f"⚠️ Prefetch error {video_path}: {e}")
                self.prefetch_queue.put({
                    'video_path': video_path,
                    'output_path': output_path,
                    'frames': [],
                    'error': str(e)
                })

        self.prefetch_queue.put(None)

    def start(self):
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def get_next(self) -> Optional[Dict]:
        return self.prefetch_queue.get()

    def stop(self):
        self.stop_flag.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)


# ============================================================
# ViTPose inference engine
# ============================================================
class ViTPoseInference:
    """ViTPose inference wrapper"""

    def __init__(self, model_path: str, yolo_path: str, device: str = None):
        self.model_path = model_path
        self.yolo_path = yolo_path
        self.device = device
        self.model = None
        self.loaded = False

        self._load_model()

    def _load_model(self):
        """Load ViTPose model"""
        try:
            print(f"🔄 Loading ViTPose model...")

            self.model = VitInference(
                model=self.model_path,
                yolo=self.yolo_path,
                model_name=VITPOSE_MODEL_SIZE,
                dataset=VITPOSE_DATASET,
                yolo_size=YOLO_SIZE,
                device=self.device,
                is_video=True,
                single_pose=False,
                yolo_step=1
            )

            self.loaded = True
            print(f"✅ ViTPose loaded on {self.model.device}")

        except Exception as e:
            print(f"❌ Failed to load ViTPose: {e}")
            import traceback
            traceback.print_exc()

    def reset(self):
        """Reset tracker (called for each new video)"""
        if self.model:
            self.model.reset()

    def inference(self, frame: np.ndarray) -> Dict:
        """Run inference on a single frame"""
        if not self.loaded:
            return {}
        return self.model.inference(frame)

    def process_video_frames(self, frames: List[np.ndarray],
                            show_progress: bool = False) -> List[Dict]:
        """Process all frames of a video"""
        if not self.loaded or not frames:
            return []

        self.reset()

        results = []

        if show_progress and TQDM_AVAILABLE:
            frame_iter = tqdm(frames, desc="  Frames", leave=False)
        else:
            frame_iter = frames

        for frame in frame_iter:
            frame_result = self.inference(frame)
            results.append(frame_result)

        return results


# ============================================================
# COCO-WholeBody JSON exporter
# ============================================================
class COCOWholeBodyExporter:
    """Export JSON in COCO-WholeBody format"""

    @staticmethod
    def convert_keypoints_to_coco(keypoints: np.ndarray,
                                   confidence_threshold: float = 0.3) -> List:
        """Convert keypoints to COCO format
        
        easy_ViTPose output format: (y, x, score)
        COCO standard format: (x, y, visibility)
        """
        coco_keypoints = []

        for kp in keypoints:
            y, x, conf = kp
            # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
            v = 2 if conf > confidence_threshold else (1 if conf > 0.1 else 0)
            # COCO format is (x, y, v), note the order!
            coco_keypoints.extend([float(x), float(y), int(v)])

        return coco_keypoints

    @staticmethod
    def export(frames_data: List[Dict],
               output_path: str,
               video_path: str,
               width: int,
               height: int,
               fps: float = 30.0) -> bool:
        """Export as JSON in COCO-WholeBody format"""
        try:
            images = []
            annotations = []
            ann_id = 1

            video_name = os.path.basename(video_path)

            for frame_id, frame_keypoints in enumerate(frames_data):
                images.append({
                    "id": frame_id,
                    "file_name": f"{video_name}_frame_{frame_id:06d}.jpg",
                    "width": width,
                    "height": height,
                    "frame_id": frame_id,
                    "video_name": video_name
                })

                for person_id, keypoints in frame_keypoints.items():
                    if isinstance(keypoints, np.ndarray):
                        kp_array = keypoints
                    else:
                        kp_array = np.array(keypoints)

                    coco_kp = COCOWholeBodyExporter.convert_keypoints_to_coco(kp_array)

                    valid_kp = kp_array[kp_array[:, 2] > 0.3]
                    if len(valid_kp) > 0:
                        x_coords = valid_kp[:, 1]
                        y_coords = valid_kp[:, 0]
                        x1, x2 = float(x_coords.min()), float(x_coords.max())
                        y1, y2 = float(y_coords.min()), float(y_coords.max())
                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        bbox = [x1, y1, bbox_w, bbox_h]
                        area = bbox_w * bbox_h
                    else:
                        bbox = [0, 0, 0, 0]
                        area = 0

                    num_keypoints = int(np.sum(kp_array[:, 2] > 0.3))

                    annotations.append({
                        "id": ann_id,
                        "image_id": frame_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "keypoints": coco_kp,
                        "num_keypoints": num_keypoints,
                        "person_id": int(person_id) if isinstance(person_id, (int, np.integer)) else person_id
                    })
                    ann_id += 1

            coco_output = {
                "info": {
                    "description": "COCO-WholeBody 133 Keypoints from ViTPose",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "ViTPose Batch Processor",
                    "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_video": video_path,
                    "fps": fps
                },
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [{
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_WHOLEBODY_KEYPOINTS,
                    "skeleton": COCO_WHOLEBODY_SKELETON
                }]
            }

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_output, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"❌ Failed to export JSON: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================
# Main processor
# ============================================================
class BatchVideoProcessor:
    """Batch video processor"""

    def __init__(self, vitpose_path: str, yolo_path: str, device: str = None):
        self.vitpose = ViTPoseInference(
            model_path=vitpose_path,
            yolo_path=yolo_path,
            device=device
        )
        self.exporter = COCOWholeBodyExporter()

    def process_single_video(self,
                             video_path: str,
                             output_path: str,
                             frames: List[np.ndarray] = None,
                             width: int = None,
                             height: int = None,
                             fps: float = 30.0,
                             show_progress: bool = False) -> bool:
        """Process a single video"""

        if frames is None:
            reader = VideoReaderFactory.create(video_path)
            frames = reader.read_all()
            width = reader.width
            height = reader.height
            fps = reader.fps
            reader.close()

        if not frames:
            print(f"⚠️ No frames read from {video_path}")
            return False

        results = self.vitpose.process_video_frames(frames, show_progress=show_progress)

        success = self.exporter.export(
            frames_data=results,
            output_path=output_path,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps
        )

        return success

    def process_video_streaming(self,
                                video_path: str,
                                output_path: str,
                                show_progress: bool = False) -> bool:
        """Stream video processing - read and process frame by frame to reduce memory usage"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Cannot open video: {video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Cannot get frame count, use regular processing
            cap.release()
            return self.process_single_video(video_path, output_path, show_progress=show_progress)
        
        # Reset tracker
        self.vitpose.reset()
        
        results = []
        
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=total_frames, desc="  Frames", leave=False)
        else:
            pbar = None
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference
            frame_result = self.vitpose.inference(frame_rgb)
            results.append(frame_result)
            
            frame_count += 1
            if pbar:
                pbar.update(1)
            
            # Release frame memory
            del frame, frame_rgb
        
        if pbar:
            pbar.close()
        
        cap.release()
        
        if not results:
            print(f"⚠️ No frames processed from {video_path}")
            return False
        
        # Export results
        success = self.exporter.export(
            frames_data=results,
            output_path=output_path,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps
        )
        
        # Cleanup
        del results
        
        return success


def find_videos(base_path: str) -> List[Tuple[str, str]]:
    """Find all video files"""
    videos = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_path)
                videos.append((path, rel))
    return videos


def check_existing(videos: List[Tuple[str, str]], output_base: str) -> List[Tuple[str, str]]:
    """Check already processed videos"""
    pending = []
    for path, rel in videos:
        out = os.path.join(output_base, os.path.splitext(rel)[0] + '.json')
        if not os.path.exists(out):
            pending.append((path, rel))
    return pending


def print_system_info():
    """Print system information"""
    print("\n" + "=" * 50)
    print("📋 System Information")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"CUDA memory: {gpu_mem:.1f} GB")
    print(f"Decord available: {DECORD_AVAILABLE}")
    print("=" * 50)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🚀 ViTPose Batch Video Processor")
    print("   COCO-WholeBody 133 Keypoints")
    print("   With Auto Model Download")
    print("=" * 60)

    # Print system information
    print_system_info()

    # Set up models (auto-download)
    vitpose_path, yolo_path = setup_models()
    if vitpose_path is None:
        print("\n❌ Failed to setup models. Exiting.")
        return

    # Check input path
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"\n❌ Input path not found: {INPUT_VIDEO_PATH}")
        print("   Please set INPUT_VIDEO_PATH to your video directory.")
        return

    # Scan videos
    print(f"\n🔍 Scanning {INPUT_VIDEO_PATH}...")
    all_videos = find_videos(INPUT_VIDEO_PATH)
    print(f"📊 Found {len(all_videos)} videos")

    if not all_videos:
        print("No videos found!")
        return

    # Check completed items
    os.makedirs(OUTPUT_JSON_PATH, exist_ok=True)
    pending = check_existing(all_videos, OUTPUT_JSON_PATH)

    completed_before = len(all_videos) - len(pending)
    if len(all_videos) > 0:
        print(f"📊 Already done: {completed_before}/{len(all_videos)} ({100*completed_before/len(all_videos):.1f}%)")
    print(f"📋 Remaining: {len(pending)}")

    if not pending:
        print("🎉 All done!")
        return

    # Prepare task list
    video_list = [(p, os.path.join(OUTPUT_JSON_PATH, os.path.splitext(r)[0] + '.json'))
                  for p, r in pending]

    # Initialize processor
    processor = BatchVideoProcessor(vitpose_path, yolo_path)

    if not processor.vitpose.loaded:
        print("❌ Failed to load models")
        return

    # Processing loop - use streaming mode to reduce memory
    print(f"\n🚀 Starting processing (streaming mode for low memory usage)...")
    
    start_time = time.time()
    completed, failed = 0, 0

    if TQDM_AVAILABLE:
        pbar = tqdm(total=len(video_list), desc="Processing videos")
    else:
        pbar = None
        print(f"Processing {len(video_list)} videos...")

    for video_path, output_path in video_list:
        try:
            t0 = time.time()

            # Use streaming processing
            success = processor.process_video_streaming(
                video_path=video_path,
                output_path=output_path,
                show_progress=False
            )

            if success:
                completed += 1
                elapsed = max(time.time() - t0, 0.01)
                if pbar:
                    pbar.set_postfix({"time": f"{elapsed:.1f}s", "✅": completed, "❌": failed})
                else:
                    print(f"  ✅ {os.path.basename(video_path)} ({elapsed:.1f}s)")
            else:
                failed += 1
                if pbar:
                    pbar.set_postfix({"✅": completed, "❌": failed})

        except Exception as e:
            print(f"❌ Error {os.path.basename(video_path)}: {e}")
            failed += 1

        if pbar:
            pbar.update(1)

        # Periodically clear memory
        if (completed + failed) % CLEAR_MEMORY_INTERVAL == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if pbar:
        pbar.close()

    # Statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 Done!")
    print(f"⏱️ Time: {total_time/3600:.2f}h ({total_time:.1f}s)")
    print(f"📊 Completed: {completed}, Failed: {failed}")
    if total_time > 0:
        print(f"📈 Rate: {(completed+failed)/total_time*3600:.1f} videos/hour")


if __name__ == "__main__":
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        main()

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
