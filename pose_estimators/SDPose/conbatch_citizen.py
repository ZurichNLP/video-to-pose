#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDPose Fast Batch Video Processor - Final Optimized Version
============================================================
- webm: cv2 multi-process decoding + prefetch queue (parallel read/inference)
- mp4/avi/mov/mkv: Decord full-batch decoding
- Optimized for L4 GPU (22GB) + 4 CPU workers
- OOM protection + automatic memory management
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import json
import time
import queue
import threading
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

# Decord
try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("native")
    DECORD_AVAILABLE = True
    print("✅ Decord available")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Decord not available")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from models.HeatmapHead import get_heatmap_head
from models.ModifiedUNet import Modified_forward
from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

try:
    from diffusers.utils import is_xformers_available
except ImportError:
    def is_xformers_available():
        return False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available")

# ============================================================
# Configuration
# ============================================================
REMOTE_BASE_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/ASL_Citizen/videos/"
LOCAL_OUTPUT_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/ASL_Citizen/sdpose/"
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
MODEL_REPO = "teemosliang/SDPose-Wholebody"
DEFAULT_YOLO_MODEL = "yolo11x.pt"

# L4 GPU + 4 CPU optimization
NUM_CPU_WORKERS = 4
BATCH_SIZE = 4  # SDPose uses about 15GB, leaving about 7GB
YOLO_CONFIDENCE = 0.5
ENABLE_FP16 = True
DECORD_NUM_THREADS = 4
MAX_FRAMES_IN_MEMORY = 500

# Format groups
DECORD_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
CV2_FORMATS = {'.webm', '.flv', '.wmv'}

# Prefetch queue size
PREFETCH_QUEUE_SIZE = 3


# ============================================================
# Multi-process cv2 decode function (must be defined at top level)
# ============================================================
def _cv2_decode_segment(args):
    """Decode one video segment in a single worker."""
    video_path, start_frame, end_frame = args
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return start_frame, []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return start_frame, frames


def _cv2_read_all_sequential(video_path):
    """Read all frames sequentially."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ============================================================
# Video readers
# ============================================================
class VideoReaderFactory:
    """Create the optimal reader based on video format."""
    
    @staticmethod
    def create(video_path: str, num_workers: int = NUM_CPU_WORKERS):
        ext = os.path.splitext(video_path)[1].lower()
        
        if ext in DECORD_FORMATS and DECORD_AVAILABLE:
            return DecordReader(video_path)
        else:
            return CV2ParallelReader(video_path, num_workers)


class DecordReader:
    """
    Decord reader - suitable for mp4/avi/mov/mkv.
    Decodes the full video in one batch.
    """
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
            print(f"⚠️ Decord failed: {e}")
            self._fallback_cv2()
    
    def _fallback_cv2(self):
        """Use cv2 as fallback when Decord fails."""
        self.vr = None
        self.ready = False
        
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Count frames
            self.total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                self.total_frames += 1
        cap.release()
    
    def read_all(self) -> List[np.ndarray]:
        """Read all frames at once."""
        if self.total_frames <= 0:
            return []
        
        if self.ready and self.vr is not None:
            try:
                indices = list(range(self.total_frames))
                frames_rgb = self.vr.get_batch(indices)
                
                if hasattr(frames_rgb, 'asnumpy'):
                    frames_rgb = frames_rgb.asnumpy()
                
                # RGB -> BGR
                return [cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR) 
                        for i in range(len(frames_rgb))]
            except Exception as e:
                print(f"⚠️ Decord batch failed: {e}")
        
        # Fallback
        return _cv2_read_all_sequential(self.video_path)
    
    def close(self):
        self.vr = None


class CV2ParallelReader:
    """
    cv2 parallel reader - suitable for webm.
    Supports prefetch to overlap reading and inference.
    """
    def __init__(self, video_path: str, num_workers: int = NUM_CPU_WORKERS):
        self.video_path = video_path
        self.num_workers = num_workers
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 30.0
        
        self._get_info()
    
    def _get_info(self):
        """Collect video metadata and count frames."""
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # webm frame count may be inaccurate; count manually
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                frame_count = 0
                while cap.read()[0]:
                    frame_count += 1
            self.total_frames = frame_count
        cap.release()
    
    def read_all(self) -> List[np.ndarray]:
        """Read all frames in parallel."""
        if self.total_frames <= 0:
            return []
        
        # Read sequentially when frame count is small
        if self.total_frames < 50:
            return _cv2_read_all_sequential(self.video_path)

        # Split decode tasks
        frames_per_worker = self.total_frames // self.num_workers
        tasks = []
        for i in range(self.num_workers):
            start = i * frames_per_worker
            end = (i + 1) * frames_per_worker if i < self.num_workers - 1 else self.total_frames
            tasks.append((self.video_path, start, end))
        
        # Parallel decode with threads (more stable than ProcessPoolExecutor in some environments)
        results = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_cv2_decode_segment, task) for task in tasks]
            for future in futures:
                start_frame, frames = future.result()
                results[start_frame] = frames
        
        # Merge in chronological order
        all_frames = []
        for start in sorted(results.keys()):
            all_frames.extend(results[start])
        
        return all_frames
    
    def close(self):
        pass


class PrefetchVideoProcessor:
    """
    Video processor with prefetch.
    A background thread reads the next video while current inference is running.
    """
    def __init__(self, video_list: List[Tuple[str, str]], num_workers: int = NUM_CPU_WORKERS):
        self.video_list = video_list
        self.num_workers = num_workers
        self.prefetch_queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
        self.stop_flag = threading.Event()
        self.prefetch_thread = None
    
    def _prefetch_worker(self):
        """Background prefetch worker."""
        for video_path, output_path in self.video_list:
            if self.stop_flag.is_set():
                break
            
            try:
                reader = VideoReaderFactory.create(video_path, self.num_workers)
                frames = reader.read_all()
                info = {
                    'video_path': video_path,
                    'output_path': output_path,
                    'frames': frames,
                    'width': reader.width,
                    'height': reader.height,
                    'total_frames': len(frames)
                }
                reader.close()
                
                # Put into queue (blocks until there is free space)
                self.prefetch_queue.put(info)
                
            except Exception as e:
                print(f"⚠️ Prefetch error {video_path}: {e}")
                self.prefetch_queue.put({
                    'video_path': video_path,
                    'output_path': output_path,
                    'frames': [],
                    'width': 0,
                    'height': 0,
                    'total_frames': 0,
                    'error': str(e)
                })
        
        # Send end signal
        self.prefetch_queue.put(None)
    
    def start(self):
        """Start the prefetch thread."""
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def get_next(self) -> Optional[Dict]:
        """Get the next prefetched video."""
        return self.prefetch_queue.get()
    
    def stop(self):
        """Stop prefetching."""
        self.stop_flag.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)


# ============================================================
# YOLO detector
# ============================================================
class FastYOLODetector:
    def __init__(self, device='cuda:0', detect_width=960):
        self.device = device
        self.detect_width = detect_width
        self.model = None
        self.loaded = False
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(DEFAULT_YOLO_MODEL)
                self.model.to(self.device)
                self.loaded = True
                print(f"✅ YOLO loaded on {self.device}")
            except Exception as e:
                print(f"❌ YOLO failed: {e}")
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List]:
        if not self.loaded or not images:
            return [[[0, 0, img.shape[1], img.shape[0]]] for img in images]
        
        # Detect on resized frames
        resized, scales = [], []
        for img in images:
            h, w = img.shape[:2]
            if w > self.detect_width:
                scale = self.detect_width / w
                img = cv2.resize(img, (self.detect_width, int(h * scale)))
            else:
                scale = 1.0
            resized.append(img)
            scales.append(scale)
        
        results = self.model(resized, verbose=False, device=self.device)
        
        processed = []
        for i, (result, scale) in enumerate(zip(results, scales)):
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > YOLO_CONFIDENCE:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if scale != 1.0:
                            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
            
            if not boxes:
                h, w = images[i].shape[:2]
                boxes = [[0, 0, w, h]]
            processed.append(boxes)
        
        return processed


# ============================================================
# SDPose inference
# ============================================================
class FastSDPoseInference:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.pipeline = None
        self.loaded = False
        self.input_size = (768, 1024)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size[1], self.input_size[0])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_model(self):
        try:
            print(f"🔄 Loading SDPose on {self.device}...")
            
            cache_dir = snapshot_download(
                repo_id=MODEL_REPO,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
                cache_dir="./model_cache"
            )
            
            dtype = torch.float16 if ENABLE_FP16 else torch.float32
            
            unet = UNet2DConditionModel.from_pretrained(
                cache_dir, subfolder="unet",
                class_embed_type="projection",
                projection_class_embeddings_input_dim=4,
                torch_dtype=dtype, low_cpu_mem_usage=True
            )
            unet = Modified_forward(unet, keypoint_scheme="wholebody")
            
            vae = AutoencoderKL.from_pretrained(
                cache_dir, subfolder="vae",
                torch_dtype=dtype, low_cpu_mem_usage=True
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(cache_dir, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(
                cache_dir, subfolder="text_encoder",
                torch_dtype=dtype, low_cpu_mem_usage=True
            )
            
            hm_decoder = get_heatmap_head(mode="wholebody")
            decoder_file = os.path.join(cache_dir, "decoder", "decoder.safetensors")
            if not os.path.exists(decoder_file):
                decoder_file = os.path.join(cache_dir, "decoder.safetensors")
            hm_decoder.load_state_dict(load_file(decoder_file, device="cpu"), strict=True)
            if ENABLE_FP16:
                hm_decoder = hm_decoder.half()
            
            scheduler = DDPMScheduler.from_pretrained(cache_dir, subfolder="scheduler")
            
            unet.to(self.device)
            vae.to(self.device)
            text_encoder.to(self.device)
            hm_decoder.to(self.device)
            
            self.pipeline = SDPose_D_Pipeline(
                unet=unet, vae=vae, tokenizer=tokenizer,
                text_encoder=text_encoder, scheduler=scheduler, decoder=hm_decoder
            )
            
            if is_xformers_available():
                try:
                    self.pipeline.unet.enable_xformers_memory_efficient_attention()
                except:
                    pass
            
            self.loaded = True
            print("✅ SDPose loaded!")
            return True
            
        except Exception as e:
            print(f"❌ SDPose failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_batch(self, images: List[np.ndarray], bboxes_list: List[List]):
        tensors, metadata = [], []
        
        for img, bboxes in zip(images, bboxes_list):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            orig_size = (img.shape[1], img.shape[0])
            
            for bbox in bboxes[:1]:
                x1, y1, x2, y2 = map(int, bbox[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
                
                crop_info = (x1, y1, x2-x1, y2-y1)
                crop = pil_img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else pil_img
                
                tensors.append(self.transform(crop).unsqueeze(0))
                metadata.append({'crop_info': crop_info, 'original_size': orig_size})
        
        if tensors:
            return torch.cat(tensors, 0), metadata
        return torch.empty(0), []
    
    def process_batch(self, batch_tensor: torch.Tensor):
        if not self.loaded or batch_tensor.size(0) == 0:
            return []
        
        batch_tensor = batch_tensor.to(self.device)
        
        try:
            with torch.inference_mode():
                if ENABLE_FP16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.pipeline(
                            batch_tensor, timesteps=[999],
                            test_cfg={'flip_test': False},
                            show_progress_bar=False, mode="inference"
                        )
                else:
                    outputs = self.pipeline(
                        batch_tensor, timesteps=[999],
                        test_cfg={'flip_test': False},
                        show_progress_bar=False, mode="inference"
                    )
            
            results = []
            for out in outputs:
                k = out.keypoints[0]
                s = out.keypoint_scores[0]
                if isinstance(k, torch.Tensor):
                    k = k.detach().cpu().numpy()
                if isinstance(s, torch.Tensor):
                    s = s.detach().cpu().numpy()
                results.append({"keypoints": k, "scores": s})
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM, processing one by one...")
            torch.cuda.empty_cache()
            
            results = []
            for i in range(batch_tensor.size(0)):
                try:
                    single = batch_tensor[i:i+1]
                    with torch.inference_mode():
                        if ENABLE_FP16:
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                out = self.pipeline(
                                    single, timesteps=[999],
                                    test_cfg={'flip_test': False},
                                    show_progress_bar=False, mode="inference"
                                )[0]
                        else:
                            out = self.pipeline(
                                single, timesteps=[999],
                                test_cfg={'flip_test': False},
                                show_progress_bar=False, mode="inference"
                            )[0]
                    
                    k = out.keypoints[0]
                    s = out.keypoint_scores[0]
                    if isinstance(k, torch.Tensor):
                        k = k.detach().cpu().numpy()
                    if isinstance(s, torch.Tensor):
                        s = s.detach().cpu().numpy()
                    results.append({"keypoints": k, "scores": s})
                except:
                    results.append({"keypoints": np.zeros((133, 2)), "scores": np.zeros(133)})
                    torch.cuda.empty_cache()
            
            return results
        
        finally:
            del batch_tensor
            torch.cuda.empty_cache()
    
    def restore_keypoints(self, keypoints, crop_info, original_size):
        x1, y1, crop_w, crop_h = crop_info
        scale_x = crop_w / self.input_size[0]
        scale_y = crop_h / self.input_size[1]
        
        restored = keypoints.copy()
        restored[:, 0] = keypoints[:, 0] * scale_x + x1
        restored[:, 1] = keypoints[:, 1] * scale_y + y1
        return restored


# ============================================================
# Main processor
# ============================================================
class VideoProcessor:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.yolo = FastYOLODetector(device)
        self.sdpose = FastSDPoseInference(device)
    
    def load_models(self):
        return self.sdpose.load_model()
    
    def process_frames(self, frames: List[np.ndarray]) -> List[dict]:
        """Process a batch of frames."""
        results = []
        
        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            # YOLO
            bboxes_list = self.yolo.detect_batch(batch_frames)
            
            # SDPose
            tensor, metadata = self.sdpose.preprocess_batch(batch_frames, bboxes_list)
            pose_results = self.sdpose.process_batch(tensor)
            
            # Assemble results
            idx = 0
            for fid, bboxes in zip(batch_indices, bboxes_list):
                persons = []
                for bbox in bboxes[:1]:
                    if idx < len(pose_results) and idx < len(metadata):
                        kp = pose_results[idx]['keypoints']
                        scores = pose_results[idx]['scores']
                        kp_restored = self.sdpose.restore_keypoints(
                            kp, metadata[idx]['crop_info'], metadata[idx]['original_size']
                        )
                        persons.append({
                            "bbox": bbox,
                            "keypoints": kp_restored.tolist(),
                            "scores": scores.tolist()
                        })
                        idx += 1
                results.append({"frame_id": fid, "people": persons})
            
            del tensor, pose_results
            torch.cuda.empty_cache()
        
        return results
    
    def save_coco_json(self, frames_data: List[dict], output_path: str,
                       video_path: str, width: int, height: int) -> bool:
        try:
            images, annotations = [], []
            ann_id = 1
            
            for fr in frames_data:
                images.append({
                    "file_name": f"{os.path.basename(video_path)}_f{fr['frame_id']:06d}",
                    "height": height, "width": width, "id": fr['frame_id']
                })
                
                for p in fr["people"]:
                    x1, y1, x2, y2 = p["bbox"]
                    w, h = x2 - x1, y2 - y1
                    
                    keypoints = []
                    for (x, y), sc in zip(p["keypoints"], p["scores"]):
                        v = 2 if sc > 0.3 else 1
                        keypoints += [float(x), float(y), v]
                    
                    annotations.append({
                        "id": ann_id, "image_id": fr["frame_id"], "category_id": 1,
                        "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0)
                    })
                    ann_id += 1
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump({
                    "images": images,
                    "annotations": annotations,
                    "categories": [{
                        "id": 1, "name": "person",
                        "keypoints": [f"kpt_{i}" for i in range(133)],
                        "skeleton": []
                    }]
                }, f)
            
            return True
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False


# ============================================================
# Main function
# ============================================================
def find_videos(base_path):
    videos = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_path)
                videos.append((path, rel))
    return videos


def check_existing(videos, output_base):
    pending = []
    for path, rel in videos:
        out = os.path.join(output_base, os.path.splitext(rel)[0] + '.json')
        if not os.path.exists(out):
            pending.append((path, rel))
    return pending


def main():
    # Environment setup
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=" * 60)
    print("🚀 SDPose Batch Processor - Final Optimized")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    print(f"🔧 Config: Batch={BATCH_SIZE}, Workers={NUM_CPU_WORKERS}, FP16={ENABLE_FP16}")
    
    # Scan videos
    if not os.path.exists(REMOTE_BASE_PATH):
        print(f"❌ Path not found: {REMOTE_BASE_PATH}")
        return
    
    print(f"\n🔍 Scanning {REMOTE_BASE_PATH}...")
    all_videos = find_videos(REMOTE_BASE_PATH)
    print(f"📊 Found {len(all_videos)} videos")
    
    if not all_videos:
        return
    
    os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
    pending = check_existing(all_videos, LOCAL_OUTPUT_PATH)
    
    completed_before = len(all_videos) - len(pending)
    print(f"📊 Already done: {completed_before}/{len(all_videos)} ({100*completed_before/len(all_videos):.1f}%)")
    print(f"📋 Remaining: {len(pending)}")
    
    if not pending:
        print("🎉 All done!")
        return
    
    # Prepare task list
    video_list = [(p, os.path.join(LOCAL_OUTPUT_PATH, os.path.splitext(r)[0] + '.json')) 
                  for p, r in pending]

    # Initialize
    processor = VideoProcessor()
    if not processor.load_models():
        return
    
    # Start prefetch
    print(f"\n🚀 Starting with prefetch (queue={PREFETCH_QUEUE_SIZE})...")
    prefetcher = PrefetchVideoProcessor(video_list, NUM_CPU_WORKERS)
    prefetcher.start()
    
    # Processing loop
    start_time = time.time()
    completed, failed = 0, 0
    
    with tqdm(total=len(video_list), desc="Processing") as pbar:
        while True:
            video_info = prefetcher.get_next()
            
            if video_info is None:  # End signal
                break
            
            video_path = video_info['video_path']
            output_path = video_info['output_path']
            frames = video_info['frames']
            width = video_info['width']
            height = video_info['height']
            
            if 'error' in video_info or not frames:
                failed += 1
                pbar.update(1)
                continue
            
            try:
                # Inference
                t0 = time.time()
                results = processor.process_frames(frames)
                infer_time = time.time() - t0
                
                # Save results
                if processor.save_coco_json(results, output_path, video_path, width, height):
                    completed += 1
                    fps = len(frames) / max(infer_time, 0.01)
                    pbar.set_postfix({"fps": f"{fps:.1f}", "✅": completed, "❌": failed})
                else:
                    failed += 1
                
            except Exception as e:
                print(f"❌ Error {os.path.basename(video_path)}: {e}")
                failed += 1
            
            pbar.update(1)
            
            # Periodic cleanup
            if (completed + failed) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    prefetcher.stop()
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 Done!")
    print(f"⏱️ Time: {total_time/3600:.2f}h")
    print(f"📊 Completed: {completed}, Failed: {failed}")
    print(f"📈 Rate: {(completed+failed)/max(total_time,1)*3600:.1f} videos/hour")


if __name__ == "__main__":
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        torch.set_num_threads(NUM_CPU_WORKERS)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        mp.set_start_method('spawn', force=True)
        
        main()
        
    except Exception as e:
        print(f"CRITICAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
