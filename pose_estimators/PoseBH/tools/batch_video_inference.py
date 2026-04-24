#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViTPose Batch Video Inference - Memory Optimized Version
=========================================================
基于 PoseBH/ViTPose 的视频数据集批量推理脚本
- 使用 Decord 加速视频读取
- 流式处理，控制内存峰值
- 支持单节点和多节点分布式推理
- 输出 COCO 格式 JSON

Usage (单节点):
    python tools/batch_video_inference.py \
        --config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
        --checkpoint weights/wholebody. pth \
        --input_dir /path/to/videos \
        --output_dir /path/to/output

Usage (多节点分布式):
    # Node 0
    python -m torch.distributed.launch --nnodes 2 --node_rank 0 --nproc_per_node 1 \
        --master_addr $MASTER_ADDR --master_port 23459 \
        tools/batch_video_inference.py \
        --config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
        --checkpoint weights/wholebody.pth \
        --input_dir /path/to/videos \
        --output_dir /path/to/output \
        --launcher pytorch

    # Node 1
    python -m torch.distributed. launch --nnodes 2 --node_rank 1 --nproc_per_node 1 \
        --master_addr $MASTER_ADDR --master_port 23459 \
        tools/batch_video_inference.py \
        --config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
        --checkpoint weights/wholebody.pth \
        --input_dir /path/to/videos \
        --output_dir /path/to/output \
        --launcher pytorch
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.distributed as dist
import json
import time
import gc
import warnings
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional, Tuple, List, Dict, Generator
from tqdm import tqdm

# Decord - 高效视频解码
try:
    import decord
    from decord import VideoReader, cpu, gpu
    decord. bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
    print("✅ Decord available (torch bridge)")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Decord not available, falling back to OpenCV")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmpose.datasets import DatasetInfo

# Optional:  YOLO for person detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available, using full frame as bbox")


# ============================================================
# 配置参数
# ============================================================
VIDEO_EXTENSIONS = ['.mp4', '. avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

# 内存优化配置
DEFAULT_BATCH_SIZE = 8          # 推理 batch size
DEFAULT_CHUNK_SIZE = 100        # 每次从视频读取的帧数
DECORD_NUM_THREADS = 4          # Decord 解码线程数
YOLO_CONFIDENCE = 0.5           # YOLO 人体检测置信度阈值
GC_EVERY_N_VIDEOS = 5           # 每处理 N 个视频强制 GC
ENABLE_FP16 = True              # 启用 FP16 推理


# ============================================================
# Decord 流式视频读取器
# ============================================================
class DecordVideoReader:
    """
    使用 Decord 的高效流式视频读取器
    - 支持 GPU 解码加速
    - 按需读取帧，不一次性加载整个视频
    - 自动处理各种视频格式
    """
    def __init__(self, video_path: str, num_threads: int = DECORD_NUM_THREADS, 
                 use_gpu: bool = False, gpu_id: int = 0):
        self.video_path = video_path
        self.num_threads = num_threads
        self.use_gpu = use_gpu and DECORD_AVAILABLE
        self.gpu_id = gpu_id
        
        self.vr = None
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.current_frame = 0
        
        self._init()
    
    def _init(self):
        """初始化视频读取器"""
        if DECORD_AVAILABLE:
            try:
                # 设置解码上下文
                if self.use_gpu:
                    ctx = gpu(self.gpu_id)
                else: 
                    ctx = cpu(0)
                
                self.vr = VideoReader(
                    self.video_path, 
                    ctx=ctx,
                    num_threads=self.num_threads
                )
                
                self.total_frames = len(self.vr)
                self.width = self.vr[0].shape[1]
                self.height = self.vr[0]. shape[0]
                self. fps = self.vr. get_avg_fps()
                
                return
            except Exception as e:
                print(f"⚠️ Decord failed for {self.video_path}: {e}, falling back to OpenCV")
        
        # Fallback to OpenCV
        self._init_opencv()
    
    def _init_opencv(self):
        """使用 OpenCV 作为后备"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video:  {self.video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # 获取帧数
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # webm 等格式帧数不准，需要手动计数
        ext = os.path.splitext(self. video_path)[1].lower()
        if frame_count <= 0 or ext in ['.webm', '.flv', '.wmv']:
            frame_count = 0
            while True:
                ret = self.cap.grab()
                if not ret: 
                    break
                frame_count += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self. total_frames = frame_count
        self.vr = None  # 标记使用 OpenCV
    
    def read_chunk(self, chunk_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        读取一块帧
        返回:  (帧列表 BGR格式, 帧ID列表)
        """
        frames = []
        frame_ids = []
        
        end_frame = min(self.current_frame + chunk_size, self.total_frames)
        
        if self.vr is not None:
            # 使用 Decord 批量读取
            indices = list(range(self.current_frame, end_frame))
            if not indices:
                return [], []
            
            try:
                # Decord 返回 RGB 格式的 tensor
                batch = self.vr.get_batch(indices)
                
                # 转换为 numpy BGR 格式
                if isinstance(batch, torch.Tensor):
                    batch = batch.numpy()
                
                for i, idx in enumerate(indices):
                    frame = batch[i]
                    # RGB -> BGR for consistency with OpenCV
                    frame = frame[: , :, ::-1].copy()
                    frames.append(frame)
                    frame_ids.append(idx)
                
                self.current_frame = end_frame
                
            except Exception as e:
                print(f"⚠️ Decord batch read failed: {e}")
                return [], []
        else:
            # 使用 OpenCV
            for _ in range(chunk_size):
                if self.current_frame >= self.total_frames:
                    break
                ret, frame = self.cap.read()
                if not ret: 
                    break
                frames.append(frame)
                frame_ids.append(self.current_frame)
                self.current_frame += 1
        
        return frames, frame_ids
    
    def iter_chunks(self, chunk_size: int) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """迭代器：按块读取视频"""
        while self.current_frame < self.total_frames:
            frames, frame_ids = self.read_chunk(chunk_size)
            if not frames:
                break
            yield frames, frame_ids
            gc.collect()
    
    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        self.vr = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __len__(self):
        return self. total_frames


# ============================================================
# YOLO 人体检测器
# ============================================================
class PersonDetector:
    """YOLO 人体检测器，用于获取人体 bbox"""
    
    def __init__(self, device='cuda: 0', model_path='yolov8x.pt', detect_width=960):
        self.device = device
        self.detect_width = detect_width
        self.model = None
        self.loaded = False
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.loaded = True
                print(f"✅ YOLO loaded on {self.device}")
            except Exception as e:
                print(f"⚠️ YOLO failed to load: {e}")
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List]: 
        """
        批量检测人体
        返回: List[List[bbox]], 每个图像的 bbox 列表
        """
        if not self. loaded or not images:
            # 返回整张图作为 bbox
            return [[[0, 0, img.shape[1], img.shape[0]]] for img in images]
        
        # 缩放图像用于检测
        resized = []
        scales = []
        for img in images: 
            h, w = img.shape[:2]
            if w > self.detect_width:
                scale = self.detect_width / w
                img_resized = cv2.resize(img, (self.detect_width, int(h * scale)))
            else:
                scale = 1.0
                img_resized = img
            resized.append(img_resized)
            scales.append(scale)
        
        # 批量推理
        results = self.model(resized, verbose=False, device=self.device)
        
        processed = []
        for i, (result, scale) in enumerate(zip(results, scales)):
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    # class 0 是 person
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > YOLO_CONFIDENCE: 
                        x1, y1, x2, y2 = box.xyxy[0]. cpu().numpy()
                        # 还原到原始尺寸
                        if scale != 1.0:
                            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
            
            if not boxes:
                # 如果没检测到人，使用整张图
                h, w = images[i].shape[:2]
                boxes = [[0, 0, w, h]]
            
            processed.append(boxes)
        
        return processed


# ============================================================
# ViTPose 推理器
# ============================================================
class ViTPoseInference:
    """ViTPose 模型推理封装"""
    
    def __init__(self, config_path:  str, checkpoint_path: str, device: str = 'cuda:0'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self. dataset = None
        self.dataset_info = None
        self.loaded = False
    
    def load_model(self):
        """加载模型"""
        try:
            print(f"🔄 Loading ViTPose model from {self.config_path}...")
            
            self.model = init_pose_model(
                self.config_path, 
                self.checkpoint_path, 
                device=self. device
            )
            
            # 获取数据集信息
            self.dataset = self.model.cfg. data['test']['type']
            dataset_info = self.model.cfg.data['test']. get('dataset_info', None)
            
            if dataset_info is not None:
                self.dataset_info = DatasetInfo(dataset_info)
            
            # 设置为评估模式
            self.model.eval()
            
            # 可选：启用 FP16
            if ENABLE_FP16 and 'cuda' in self.device:
                self.model = self.model.half()
            
            self.loaded = True
            print(f"✅ ViTPose loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ViTPose: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def inference_batch(self, images: List[np.ndarray], 
                       bboxes_list: List[List]) -> List[Dict]:
        """
        批量推理
        Args:
            images: BGR 格式图像列表
            bboxes_list: 每张图的 bbox 列表
        Returns:
            每张图的姿态估计结果列表
        """
        if not self.loaded:
            return []
        
        all_results = []
        
        for img, bboxes in zip(images, bboxes_list):
            # 构建 person_results
            person_results = []
            for bbox in bboxes:
                person_results.append({
                    'bbox': np.array(bbox + [1.0])  # [x1, y1, x2, y2, score]
                })
            
            try:
                # 推理
                with torch.no_grad():
                    pose_results, _ = inference_top_down_pose_model(
                        self. model,
                        img,
                        person_results,
                        format='xyxy',
                        dataset=self.dataset,
                        dataset_info=self.dataset_info,
                        return_heatmap=False,
                        outputs=None
                    )
                
                # 提取关键点
                frame_result = {
                    'people': []
                }
                
                for pose in pose_results:
                    keypoints = pose['keypoints']  # (N, 3) - x, y, score
                    bbox = pose. get('bbox', bboxes[0] if bboxes else [0, 0, img.shape[1], img.shape[0]])
                    
                    frame_result['people'].append({
                        'bbox': bbox[: 4]. tolist() if isinstance(bbox, np.ndarray) else bbox[: 4],
                        'keypoints': keypoints[: , :2].tolist(),
                        'scores': keypoints[:, 2]. tolist()
                    })
                
                all_results.append(frame_result)
                
            except Exception as e:
                print(f"⚠️ Inference error: {e}")
                all_results.append({'people': []})
        
        return all_results


# ============================================================
# 视频处理器
# ============================================================
class VideoProcessor:
    """视频批量处理器"""
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 device: str = 'cuda:0', batch_size: int = DEFAULT_BATCH_SIZE,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 use_detector: bool = True):
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_detector = use_detector
        
        # 初始化模型
        self.pose_model = ViTPoseInference(config_path, checkpoint_path, device)
        self.detector = PersonDetector(device) if use_detector else None
    
    def load_models(self) -> bool:
        """加载所有模型"""
        return self.pose_model.load_model()
    
    def process_chunk(self, frames: List[np.ndarray], 
                     frame_ids: List[int]) -> List[Dict]:
        """处理一个 chunk 的帧"""
        results = []
        
        # 按 batch_size 分批处理
        for batch_start in range(0, len(frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_frame_ids = frame_ids[batch_start:batch_end]
            
            # 人体检测
            if self. detector and self.detector.loaded:
                bboxes_list = self.detector.detect_batch(batch_frames)
            else:
                # 使用整张图作为 bbox
                bboxes_list = [[[0, 0, f.shape[1], f.shape[0]]] for f in batch_frames]
            
            # 姿态估计
            pose_results = self.pose_model.inference_batch(batch_frames, bboxes_list)
            
            # 组装结果
            for fid, pose_result in zip(batch_frame_ids, pose_results):
                results.append({
                    'frame_id': fid,
                    'people': pose_result. get('people', [])
                })
            
            # 清理
            del batch_frames
        
        return results
    
    def process_video(self, video_path: str, output_path: str) -> bool:
        """
        处理单个视频
        流式处理：每次只读取 chunk_size 帧
        """
        try: 
            with DecordVideoReader(video_path) as reader:
                if reader.total_frames == 0:
                    print(f"⚠️ Empty video: {video_path}")
                    return False
                
                width = reader.width
                height = reader.height
                fps = reader.fps
                total_frames = reader.total_frames
                
                all_results = []
                
                # 流式处理每个 chunk
                for frames, frame_ids in reader.iter_chunks(self.chunk_size):
                    chunk_results = self.process_chunk(frames, frame_ids)
                    all_results.extend(chunk_results)
                    
                    # 释放内存
                    del frames
                    gc.collect()
                
                # 保存结果
                success = self.save_coco_json(
                    all_results, output_path, video_path,
                    width, height, fps
                )
                
                return success
                
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def save_coco_json(self, frames_data: List[Dict], output_path: str,
                      video_path: str, width:  int, height: int, fps:  float) -> bool:
        """保存为 COCO 格式 JSON"""
        try:
            # 获取关键点数量
            num_keypoints = 133  # wholebody 默认 133 个关键点
            if frames_data and frames_data[0]['people']: 
                num_keypoints = len(frames_data[0]['people'][0]['keypoints'])
            
            images = []
            annotations = []
            ann_id = 1
            
            for fr in frames_data:
                frame_id = fr['frame_id']
                
                images.append({
                    "file_name": f"{os.path.basename(video_path)}_f{frame_id: 06d}",
                    "height": height,
                    "width": width,
                    "id": frame_id
                })
                
                for person in fr['people']:
                    bbox = person['bbox']
                    x1, y1, x2, y2 = bbox[: 4]
                    w, h = x2 - x1, y2 - y1
                    
                    # 构建 COCO 格式关键点 [x, y, v, x, y, v, ...]
                    keypoints = []
                    for (x, y), score in zip(person['keypoints'], person['scores']):
                        v = 2 if score > 0.3 else (1 if score > 0.1 else 0)
                        keypoints.extend([float(x), float(y), v])
                    
                    annotations.append({
                        "id": ann_id,
                        "image_id": frame_id,
                        "category_id": 1,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": sum(1 for v in keypoints[2:: 3] if v > 0)
                    })
                    ann_id += 1
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 构建 COCO 格式 JSON
            coco_output = {
                "info": {
                    "description":  f"Pose estimation for {os.path.basename(video_path)}",
                    "video_path": video_path,
                    "fps": fps,
                    "total_frames": len(frames_data)
                },
                "images":  images,
                "annotations": annotations,
                "categories": [{
                    "id": 1,
                    "name": "person",
                    "keypoints": [f"kpt_{i}" for i in range(num_keypoints)],
                    "skeleton": []
                }]
            }
            
            with open(output_path, "w") as f:
                json.dump(coco_output, f)
            
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False


# ============================================================
# 分布式工具函数
# ============================================================
def init_distributed(launcher:  str = 'none'):
    """初始化分布式环境"""
    if launcher == 'none':
        return 0, 1  # rank, world_size
    
    if launcher == 'pytorch':
        # PyTorch distributed launch
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 设置当前 GPU
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        
        return rank, world_size
    
    raise ValueError(f"Unknown launcher: {launcher}")


def distribute_videos(videos: List[Tuple[str, str]], rank: int, 
                     world_size: int) -> List[Tuple[str, str]]: 
    """将视频列表分配给各个进程"""
    return videos[rank::world_size]


# ============================================================
# 工具函数
# ============================================================
def find_videos(base_path: str) -> List[Tuple[str, str]]: 
    """扫描所有视频文件"""
    videos = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_path)
                videos.append((path, rel))
    return sorted(videos)


def check_existing(videos: List[Tuple[str, str]], 
                  output_base: str) -> List[Tuple[str, str]]:
    """检查已处理的视频，返回待处理列表"""
    pending = []
    for path, rel in videos:
        out = os.path.join(output_base, os.path.splitext(rel)[0] + '.json')
        if not os.path.exists(out):
            pending.append((path, rel))
    return pending


def get_memory_info() -> str:
    """获取内存使用情况"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB ({mem.percent}%)"
    except ImportError:
        return "RAM: N/A (install psutil)"


def parse_args():
    parser = ArgumentParser(description='ViTPose Batch Video Inference')
    
    # 模型配置
    parser.add_argument('--config', required=True, help='Model config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    
    # 输入输出
    parser.add_argument('--input_dir', required=True, help='Input video directory')
    parser.add_argument('--output_dir', required=True, help='Output JSON directory')
    
    # 处理参数
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size for inference')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help='Number of frames to read at once')
    parser.add_argument('--no_detector', action='store_true',
                       help='Disable YOLO person detector (use full frame)')
    
    # 分布式配置
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                       help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    # 设备配置
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    
    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()
    
    # 环境设置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments: True'
    
    # 初始化分布式
    rank, world_size = init_distributed(args.launcher)
    
    # 设置设备
    if args.launcher == 'pytorch':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = f'cuda:{local_rank}'
    else:
        device = args.device
    
    # 只在 rank 0 打印信息
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 60)
        print("🚀 ViTPose Batch Video Inference")
        print("=" * 60)
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🖥️ GPU:  {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
        
        print(f"🔧 Config:  Batch={args.batch_size}, Chunk={args.chunk_size}")
        print(f"🌐 Distributed: rank={rank}, world_size={world_size}")
        print(f"📊 {get_memory_info()}")
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # 扫描视频
    if is_main:
        print(f"\n🔍 Scanning {args. input_dir}...")
    
    all_videos = find_videos(args.input_dir)
    
    if is_main:
        print(f"📊 Found {len(all_videos)} videos")
    
    if not all_videos:
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查已处理的视频
    pending = check_existing(all_videos, args.output_dir)
    
    if is_main: 
        completed_before = len(all_videos) - len(pending)
        print(f"📊 Already done: {completed_before}/{len(all_videos)}")
        print(f"📋 Remaining: {len(pending)}")
    
    if not pending:
        if is_main:
            print("🎉 All videos already processed!")
        return
    
    # 分布式：分配视频给各个进程
    my_videos = distribute_videos(pending, rank, world_size)
    
    if is_main: 
        print(f"📋 This process will handle:  {len(my_videos)} videos")
    
    # 初始化处理器
    processor = VideoProcessor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        use_detector=not args.no_detector
    )
    
    if not processor.load_models():
        return
    
    # 处理视频
    if is_main:
        print(f"\n🚀 Starting processing...")
    
    start_time = time.time()
    completed, failed = 0, 0
    
    pbar = tqdm(total=len(my_videos), desc=f"[Rank {rank}] Processing", 
                disable=not is_main)
    
    for i, (video_path, rel_path) in enumerate(my_videos):
        output_path = os.path.join(args.output_dir, 
                                   os.path.splitext(rel_path)[0] + '.json')
        
        t0 = time.time()
        success = processor.process_video(video_path, output_path)
        elapsed = time.time() - t0
        
        if success: 
            completed += 1
            pbar.set_postfix({
                "time": f"{elapsed:.1f}s",
                "✅":  completed,
                "❌": failed
            })
        else:
            failed += 1
        
        pbar.update(1)
        
        # 定期清理内存
        if (i + 1) % GC_EVERY_N_VIDEOS == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    
    # 统计
    total_time = time.time() - start_time
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"🏁 Done!")
        print(f"⏱️ Time: {total_time/3600:.2f}h")
        print(f"📊 Completed: {completed}, Failed: {failed}")
        print(f"📈 Rate: {(completed+failed)/max(total_time,1)*3600:.1f} videos/hour")
        print(f"📊 Final {get_memory_info()}")
    
    # 分布式清理
    if args.launcher == 'pytorch': 
        dist.destroy_process_group()


if __name__ == "__main__": 
    print(f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        main()
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback. print_exc()
        sys.exit(1)