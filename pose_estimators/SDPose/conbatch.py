#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDPose Batch Video Processor - Memory Optimized Version
========================================================
Addresses CPU memory OOM issues:
- Stream frames on demand instead of loading full videos into memory
- Process video in chunks to control peak memory usage
- Remove prefetch queues to avoid multi-video memory pressure
- Optimized for L4 GPU (22GB) with limited CPU memory
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
import time
import gc
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Generator
from tqdm import tqdm

# Decord - used to fetch video metadata without loading all frames at once
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
# Configuration - memory-optimized version
# ============================================================
REMOTE_BASE_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/"
LOCAL_OUTPUT_PATH = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/SDpose/"
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
MODEL_REPO = "teemosliang/SDPose-Wholebody"
DEFAULT_YOLO_MODEL = "yolo11x.pt"

# Memory optimization settings
BATCH_SIZE = 4              # SDPose inference batch size
CHUNK_SIZE = 50             # Number of frames read per chunk (key parameter)
YOLO_CONFIDENCE = 0.5
ENABLE_FP16 = True
DECORD_NUM_THREADS = 2      # Reduce Decord thread count

# Memory cleanup frequency
GC_EVERY_N_VIDEOS = 5       # Force GC every N processed videos


# ============================================================
# Streaming video reader
# ============================================================
class StreamingVideoReader:
    """
    Streaming video reader - reads frames on demand without loading whole video
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.current_frame = 0
        
        self._init()
    
    def _init(self):
        """Initialize and collect video metadata."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Get frame count
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame count may be inaccurate for webm/flv/wmv; count manually
        ext = os.path.splitext(self.video_path)[1].lower()
        if frame_count <= 0 or ext in ['.webm', '.flv', '.wmv']:
            # Fast counting
            frame_count = 0
            while True:
                ret = self.cap.grab()  # `grab` is faster than `read`
                if not ret:
                    break
                frame_count += 1
            
            # Reset to the beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.total_frames = frame_count
    
    def read_chunk(self, chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read one chunk of frames.
        Returns: (frame list, frame ID list)
        """
        frames = []
        frame_ids = []
        
        for _ in range(chunk_size):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_ids.append(self.current_frame)
            self.current_frame += 1
        
        return frames, frame_ids
    
    def iter_chunks(self, chunk_size: int) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """
        Iterator: read video chunk by chunk.
        """
        while self.current_frame < self.total_frames:
            frames, frame_ids = self.read_chunk(chunk_size)
            if not frames:
                break
            yield frames, frame_ids
            
            # Clean up after each chunk
            gc.collect()
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


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
            print("⚠️ GPU OOM, processing one by one...")
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
# Main processor - streaming version
# ============================================================
class StreamingVideoProcessor:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.yolo = FastYOLODetector(device)
        self.sdpose = FastSDPoseInference(device)
    
    def load_models(self):
        return self.sdpose.load_model()
    
    def process_chunk(self, frames: List[np.ndarray], frame_ids: List[int]) -> List[dict]:
        """Process one chunk of frames."""
        results = []
        
        # Process mini-batches by BATCH_SIZE
        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_frame_ids = frame_ids[batch_start:batch_end]
            
            # YOLO detection
            bboxes_list = self.yolo.detect_batch(batch_frames)
            
            # SDPose preprocessing
            tensor, metadata = self.sdpose.preprocess_batch(batch_frames, bboxes_list)
            
            # SDPose inference
            pose_results = self.sdpose.process_batch(tensor)
            
            # Assemble results
            idx = 0
            for fid, bboxes in zip(batch_frame_ids, bboxes_list):
                persons = []
                for bbox in bboxes[:1]:  # Keep only the first detected person
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
            
            # Clean up this batch
            del tensor, pose_results, batch_frames
        
        return results
    
    def process_video_streaming(self, video_path: str, output_path: str) -> bool:
        """
        Stream-process a single video.
        Key idea: read only CHUNK_SIZE frames at a time and release memory after processing.
        """
        try:
            with StreamingVideoReader(video_path) as reader:
                if reader.total_frames == 0:
                    print(f"⚠️ Empty video: {video_path}")
                    return False
                
                width = reader.width
                height = reader.height
                total_frames = reader.total_frames
                
                all_results = []
                
                # Stream-process each chunk
                for frames, frame_ids in reader.iter_chunks(CHUNK_SIZE):
                    chunk_results = self.process_chunk(frames, frame_ids)
                    all_results.extend(chunk_results)
                    
                    # Release this chunk's frames immediately
                    del frames
                    gc.collect()
                
                # Save results
                success = self.save_coco_json(
                    all_results, output_path, video_path, width, height
                )
                
                return success
                
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Ensure cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
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
# Utility functions
# ============================================================
def find_videos(base_path):
    """Scan all video files."""
    videos = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_path)
                videos.append((path, rel))
    return videos


def check_existing(videos, output_base):
    """Check already processed videos."""
    pending = []
    for path, rel in videos:
        out = os.path.join(output_base, os.path.splitext(rel)[0] + '.json')
        if not os.path.exists(out):
            pending.append((path, rel))
    return pending


def get_memory_info():
    """Get current memory usage."""
    import psutil
    mem = psutil.virtual_memory()
    return f"RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB ({mem.percent}%)"


# ============================================================
# Main function
# ============================================================
def main():
    # Environment setup
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=" * 60)
    print("🚀 SDPose Batch Processor - Memory Optimized")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    print(f"🔧 Config: Batch={BATCH_SIZE}, Chunk={CHUNK_SIZE}, FP16={ENABLE_FP16}")
    print(f"📊 {get_memory_info()}")
    
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
    
    # Initialize processor
    processor = StreamingVideoProcessor()
    if not processor.load_models():
        return
    
    # Processing loop
    print(f"\n🚀 Starting streaming processing...")
    start_time = time.time()
    completed, failed = 0, 0
    
    with tqdm(total=len(pending), desc="Processing") as pbar:
        for i, (video_path, rel_path) in enumerate(pending):
            output_path = os.path.join(LOCAL_OUTPUT_PATH, os.path.splitext(rel_path)[0] + '.json')
            
            t0 = time.time()
            success = processor.process_video_streaming(video_path, output_path)
            elapsed = time.time() - t0
            
            if success:
                completed += 1
                pbar.set_postfix({
                    "time": f"{elapsed:.1f}s",
                    "✅": completed,
                    "❌": failed
                })
            else:
                failed += 1
            
            pbar.update(1)
            
            # Force periodic memory cleanup
            if (i + 1) % GC_EVERY_N_VIDEOS == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
                # Print memory status every 50 videos
                if (i + 1) % 50 == 0:
                    tqdm.write(f"📊 [{i+1}/{len(pending)}] {get_memory_info()}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 Done!")
    print(f"⏱️ Time: {total_time/3600:.2f}h")
    print(f"📊 Completed: {completed}, Failed: {failed}")
    print(f"📈 Rate: {(completed+failed)/max(total_time,1)*3600:.1f} videos/hour")
    print(f"📊 Final {get_memory_info()}")


if __name__ == "__main__":
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        main()
        
    except Exception as e:
        print(f"CRITICAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
