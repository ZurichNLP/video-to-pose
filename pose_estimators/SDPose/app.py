#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDPose Gradio Space
Author: T. S. Liang
Features:
- Support both body (17 keypoints) and wholebody (133 keypoints)
- Support image and video inference
"""

# CRITICAL: Import spaces FIRST before any CUDA-related packages (torch, diffusers, etc.)
import os
import sys

# Try to import zero_gpu BEFORE any other imports
try:
    import spaces
    SPACES_ZERO_GPU = True
    print("✅ spaces (zero_gpu) imported successfully")
except ImportError:
    SPACES_ZERO_GPU = False
    print("⚠️  spaces not available, zero_gpu disabled")
    # Create dummy decorator
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Now import other packages (after spaces is imported)
import gradio as gr
import cv2
import numpy as np
import torch
import math
import json
import matplotlib.colors
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List
import tempfile
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from models.HeatmapHead import get_heatmap_head
from models.ModifiedUNet import Modified_forward
from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline
from safetensors.torch import load_file

try:
    from diffusers.utils import is_xformers_available
except ImportError:
    def is_xformers_available():
        return False

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not available, YOLO detection will be disabled")

# Constants for Gradio Space
MODEL_REPOS = {
    "body": "teemosliang/SDPose-Body",
    "wholebody": "teemosliang/SDPose-Wholebody"
}
DEFAULT_YOLO_MODEL = "yolov8n.pt"  # Will auto-download


def draw_body17_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    """
    Draw body keypoints in DWPose style (from util.py draw_bodypose)
    This function converts COCO17 format to OpenPose 18-point format with neck
    """
    H, W, C = canvas.shape
    
    if len(keypoints) >= 7:
        neck = (keypoints[5] + keypoints[6]) / 2
        neck_score = min(scores[5], scores[6]) if scores is not None else 1.0
        
        candidate = np.zeros((18, 2))
        candidate_scores = np.zeros(18)
        
        candidate[0] = keypoints[0]
        candidate[1] = neck
        candidate[2] = keypoints[6]
        candidate[3] = keypoints[8]
        candidate[4] = keypoints[10]
        candidate[5] = keypoints[5]
        candidate[6] = keypoints[7]
        candidate[7] = keypoints[9]
        candidate[8] = keypoints[12]
        candidate[9] = keypoints[14]
        candidate[10] = keypoints[16]
        candidate[11] = keypoints[11]
        candidate[12] = keypoints[13]
        candidate[13] = keypoints[15]
        candidate[14] = keypoints[2]
        candidate[15] = keypoints[1]
        candidate[16] = keypoints[4]
        candidate[17] = keypoints[3]
        
        if scores is not None:
            candidate_scores[0] = scores[0]
            candidate_scores[1] = neck_score
            candidate_scores[2] = scores[6]
            candidate_scores[3] = scores[8]
            candidate_scores[4] = scores[10]
            candidate_scores[5] = scores[5]
            candidate_scores[6] = scores[7]
            candidate_scores[7] = scores[9]
            candidate_scores[8] = scores[12]
            candidate_scores[9] = scores[14]
            candidate_scores[10] = scores[16]
            candidate_scores[11] = scores[11]
            candidate_scores[12] = scores[13]
            candidate_scores[13] = scores[15]
            candidate_scores[14] = scores[2]
            candidate_scores[15] = scores[1]
            candidate_scores[16] = scores[4]
            candidate_scores[17] = scores[3]
    else:
        return canvas
    
    avg_size = (H + W) / 2
    stickwidth = max(1, int(avg_size / 256))
    circle_radius = max(2, int(avg_size / 192))
    
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]
    
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    for i in range(len(limbSeq)):
        index = np.array(limbSeq[i]) - 1
        if index[0] >= len(candidate) or index[1] >= len(candidate):
            continue
            
        if scores is not None:
            if candidate_scores[index[0]] < threshold or candidate_scores[index[1]] < threshold:
                continue
        
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        
        if length < 1:
            continue
            
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    for i in range(18):
        if scores is not None and candidate_scores[i] < threshold:
            continue
            
        x, y = candidate[i]
        x = int(x)
        y = int(y)
        
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
            
        cv2.circle(canvas, (int(x), int(y)), circle_radius, colors[i % len(colors)], thickness=-1)
    
    return canvas


def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6):
    """Draw wholebody keypoints in DWPose style"""
    H, W, C = canvas.shape
    
    stickwidth = 4
    
    body_limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]
    
    hand_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],      # index
        [0, 9], [9, 10], [10, 11], [11, 12], # middle
        [0, 13], [13, 14], [14, 15], [15, 16], # ring
        [0, 17], [17, 18], [18, 19], [19, 20], # pinky
    ]
    
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    # Draw body limbs
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18:
                continue
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]])
            X = np.array([keypoints[idx1][1], keypoints[idx2][1]])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            
            if length < 1:
                continue
            
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    # Draw body keypoints
    if len(keypoints) >= 18:
        for i in range(18):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    
    # Draw foot keypoints
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    
    # Draw right hand
    if len(keypoints) >= 113:
        eps = 0.01
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        for i in range(92, 113):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    
    # Draw left hand
    if len(keypoints) >= 134:
        eps = 0.01
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None:
                if scores[idx1] < threshold or scores[idx2] < threshold:
                    continue
            
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    
    # Draw face keypoints
    if len(keypoints) >= 92:
        eps = 0.01
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    
    return canvas


def detect_person_yolo(image, yolo_model_path=None, confidence_threshold=0.5):
    """
    Detect person using YOLO
    Returns: List of bboxes [x1, y1, x2, y2] and whether YOLO was used
    """
    if not YOLO_AVAILABLE:
        print("⚠️  YOLO not available, using full image")
        h, w = image.shape[:2]
        return [[0, 0, w, h]], False
    
    try:
        print("🔍 Using YOLO for person detection...")
        
        # Load YOLO model
        if yolo_model_path and os.path.exists(yolo_model_path):
            print(f"   Loading custom YOLO model: {yolo_model_path}")
            model = YOLO(yolo_model_path)
        else:
            print(f"   Loading default YOLOv8n model")
            # Use default YOLOv8
            model = YOLO('yolov8n.pt')
        
        # Run detection
        print(f"   Running YOLO detection on image shape: {image.shape}")
        results = model(image, verbose=False)
        print(f"   YOLO returned {len(results)} result(s)")
        
        # Extract person detections (class 0 is person in COCO)
        person_bboxes = []
        for result in results:
            boxes = result.boxes
            print(f"   Result has {len(boxes) if boxes is not None else 0} boxes")
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0) and confidence is high enough
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    print(f"   Box: class={cls}, conf={conf:.3f}")
                    if cls == 0 and conf > confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        print(f"   ✓ Person detected: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                        person_bboxes.append([float(x1), float(y1), float(x2), float(y2), conf])
        
        if person_bboxes:
            # Sort by confidence and return all
            person_bboxes.sort(key=lambda x: x[4], reverse=True)
            bboxes = [bbox[:4] for bbox in person_bboxes]
            print(f"✅ Detected {len(bboxes)} person(s)")
            return bboxes, True
        else:
            print("⚠️  No person detected, using full image")
            h, w = image.shape[:2]
            return [[0, 0, w, h]], False
        
    except Exception as e:
        print(f"⚠️  YOLO detection failed: {e}, using full image")
        h, w = image.shape[:2]
        return [[0, 0, w, h]], False


def preprocess_image_for_sdpose(image, bbox=None, input_size=(768, 1024)):
    """Preprocess image for SDPose inference"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
        original_size = (image.shape[1], image.shape[0])
    else:
        pil_image = image
        original_size = pil_image.size
    
    crop_info = None
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(pil_image.width, x2)
        y2 = min(pil_image.height, y2)
        
        if x2 > x1 and y2 > y1:
            cropped_image = pil_image.crop((x1, y1, x2, y2))
            crop_info = (x1, y1, x2 - x1, y2 - y1)
            pil_image = cropped_image
        else:
            crop_info = (0, 0, pil_image.width, pil_image.height)
    else:
        crop_info = (0, 0, pil_image.width, pil_image.height)
    
    transform_list = [
        transforms.Resize((input_size[1], input_size[0])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    val_transform = transforms.Compose(transform_list)
    input_tensor = val_transform(pil_image).unsqueeze(0)
    
    return input_tensor, original_size, crop_info


def restore_keypoints_to_original(keypoints, crop_info, input_size, original_size):
    """Restore keypoints from cropped/resized space to original image space"""
    x1, y1, crop_w, crop_h = crop_info
    input_w, input_h = input_size
    
    scale_x = crop_w / input_w
    scale_y = crop_h / input_h
    
    keypoints_restored = keypoints.copy()
    keypoints_restored[:, 0] = keypoints[:, 0] * scale_x + x1
    keypoints_restored[:, 1] = keypoints[:, 1] * scale_y + y1
    
    return keypoints_restored


def convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body"):
    """Convert keypoints to OpenPose JSON format"""
    people = []
    
    for person_idx, (keypoints, scores) in enumerate(zip(all_keypoints, all_scores)):
        person_data = {}
        
        if keypoint_scheme == "body":
            pose_kpts = []
            for i in range(min(17, len(keypoints))):
                pose_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            
            while len(pose_kpts) < 17 * 3:
                pose_kpts.extend([0.0, 0.0, 0.0])
            
            person_data["pose_keypoints_2d"] = pose_kpts
            person_data["hand_left_keypoints_2d"] = [0.0] * 63
            person_data["hand_right_keypoints_2d"] = [0.0] * 63
            person_data["face_keypoints_2d"] = [0.0] * 204
            person_data["foot_keypoints_2d"] = [0.0] * 18
            
        else:
            # Wholebody
            pose_kpts = []
            for i in range(min(18, len(keypoints))):
                pose_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(pose_kpts) < 18 * 3:
                pose_kpts.extend([0.0, 0.0, 0.0])
            person_data["pose_keypoints_2d"] = pose_kpts
            
            foot_kpts = []
            for i in range(18, min(24, len(keypoints))):
                foot_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(foot_kpts) < 6 * 3:
                foot_kpts.extend([0.0, 0.0, 0.0])
            person_data["foot_keypoints_2d"] = foot_kpts
            
            face_kpts = []
            for i in range(24, min(92, len(keypoints))):
                face_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(face_kpts) < 68 * 3:
                face_kpts.extend([0.0, 0.0, 0.0])
            person_data["face_keypoints_2d"] = face_kpts
            
            right_hand_kpts = []
            for i in range(92, min(113, len(keypoints))):
                right_hand_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(right_hand_kpts) < 21 * 3:
                right_hand_kpts.extend([0.0, 0.0, 0.0])
            person_data["hand_right_keypoints_2d"] = right_hand_kpts
            
            left_hand_kpts = []
            for i in range(113, min(134, len(keypoints))):
                left_hand_kpts.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])])
            while len(left_hand_kpts) < 21 * 3:
                left_hand_kpts.extend([0.0, 0.0, 0.0])
            person_data["hand_left_keypoints_2d"] = left_hand_kpts
        
        people.append(person_data)
    
    result = {
        "people": people,
        "canvas_width": int(image_width),
        "canvas_height": int(image_height)
    }
    
    return result


class SDPoseInference:
    """SDPose inference class with HF Hub loading"""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.keypoint_scheme = "body"
        self.input_size = (768, 1024)
        self.model_cache_dir = None
        
    def load_model_from_hub(self, repo_id=None, keypoint_scheme="body"):
        """Load model from Hugging Face Hub"""
        try:
            if repo_id is None:
                repo_id = MODEL_REPOS.get(keypoint_scheme, MODEL_REPOS["body"])
                
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.keypoint_scheme = keypoint_scheme
            
            print(f"🔄 Loading model from: {repo_id}")
            print(f"📱 Device: {self.device}")
            
            # Download model from HF Hub
            cache_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
                cache_dir="./model_cache"
            )
            self.model_cache_dir = cache_dir
            print(f"✅ Model cached at: {cache_dir}")
            
            # Load components
            print("🔧 Loading UNet...")
            unet_path = os.path.join(cache_dir, "unet")
            if os.path.exists(unet_path):
                unet = UNet2DConditionModel.from_pretrained(
                    unet_path,
                    class_embed_type="projection",
                    projection_class_embeddings_input_dim=4,
                )
            else:
                unet = UNet2DConditionModel.from_pretrained(
                    cache_dir, 
                    subfolder="unet",
                    class_embed_type="projection",
                    projection_class_embeddings_input_dim=4,
                )
            
            unet = Modified_forward(unet, keypoint_scheme=keypoint_scheme)
            print("✅ UNet loaded")
            
            print("🔧 Loading VAE...")
            vae_path = os.path.join(cache_dir, "vae")
            if os.path.exists(vae_path):
                vae = AutoencoderKL.from_pretrained(vae_path)
            else:
                vae = AutoencoderKL.from_pretrained(cache_dir, subfolder="vae")
            print("✅ VAE loaded")
            
            print("🔧 Loading Tokenizer...")
            tokenizer_path = os.path.join(cache_dir, "tokenizer")
            if os.path.exists(tokenizer_path):
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            else:
                tokenizer = CLIPTokenizer.from_pretrained(cache_dir, subfolder="tokenizer")
            print("✅ Tokenizer loaded")
            
            print("🔧 Loading Text Encoder...")
            text_encoder_path = os.path.join(cache_dir, "text_encoder")
            if os.path.exists(text_encoder_path):
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
            else:
                text_encoder = CLIPTextModel.from_pretrained(cache_dir, subfolder="text_encoder")
            print("✅ Text Encoder loaded")
            
            print("🔧 Loading Decoder...")
            hm_decoder = get_heatmap_head(mode=keypoint_scheme)
            decoder_file = os.path.join(cache_dir, "decoder", "decoder.safetensors")
            if not os.path.exists(decoder_file):
                decoder_file = os.path.join(cache_dir, "decoder.safetensors")
            
            if os.path.exists(decoder_file):
                hm_decoder.load_state_dict(load_file(decoder_file, device="cpu"), strict=True)
                print("✅ Decoder loaded")
            else:
                print("⚠️  Decoder weights not found, using default initialization")
            
            print("🔧 Loading Scheduler...")
            scheduler_path = os.path.join(cache_dir, "scheduler")
            if os.path.exists(scheduler_path):
                noise_scheduler = DDPMScheduler.from_pretrained(scheduler_path)
            else:
                noise_scheduler = DDPMScheduler.from_pretrained(cache_dir, subfolder="scheduler")
            print("✅ Scheduler loaded")
            
            # IMPORTANT: For zero_gpu, do NOT move to GPU in main process!
            # Models will be moved to GPU inside @spaces.GPU decorated functions
            print("⚠️  Keeping models on CPU (will move to GPU during inference)")
            
            # Keep everything on CPU for now
            self.unet_cpu = unet
            self.vae_cpu = vae
            self.text_encoder_cpu = text_encoder
            self.hm_decoder_cpu = hm_decoder
            self.tokenizer = tokenizer
            self.noise_scheduler = noise_scheduler
            
            # Create pipeline on CPU
            self.pipeline = SDPose_D_Pipeline(
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=noise_scheduler,
                decoder=hm_decoder
            )
            
            # Enable xformers if available (will apply when moved to GPU)
            if is_xformers_available():
                try:
                    self.pipeline.unet.enable_xformers_memory_efficient_attention()
                    print("✅ xformers enabled")
                except:
                    pass
            
            self.model_loaded = True
            print("✅ Model loaded on CPU!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_image(self, image, enable_yolo=True, yolo_model_path=None, 
                     score_threshold=0.3, restore_coords=True, flip_test=False, process_all_persons=True, overlay_alpha=0.6):
        """
        Run inference on a single image (supports multi-person)
        overlay_alpha: Opacity of pose+black background layer (0.0=invisible, 1.0=fully opaque)
        Returns: (result_image, keypoints, scores, info_text, json_file_path)
        """
        if not self.model_loaded or self.pipeline is None:
            return None, None, None, "Model not loaded. Please load the model first.", None
        
        try:
            # Move models to GPU (only happens inside @spaces.GPU decorated function)
            if self.device.type == 'cuda' and hasattr(self, 'unet_cpu'):
                print("🚀 Moving models to GPU...")
                self.pipeline.unet = self.unet_cpu.to(self.device)
                self.pipeline.vae = self.vae_cpu.to(self.device)
                self.pipeline.text_encoder = self.text_encoder_cpu.to(self.device)
                self.pipeline.decoder = self.hm_decoder_cpu.to(self.device)
                print("✅ Models on GPU")
            
            # Handle image format: Gradio Image(type="numpy") returns RGB numpy array
            if isinstance(image, np.ndarray):
                original_image_rgb = image.copy()
            else:
                original_image_rgb = np.array(image)
            
            # Convert to BGR for YOLO (YOLO expects BGR)
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            
            # Step 1: Person detection (if enabled)
            bboxes_list = []
            detection_info = ""
            if enable_yolo:
                print(f"🔍 YOLO detection enabled (yolo_model_path: {yolo_model_path})")
                bboxes, used_yolo = detect_person_yolo(original_image_bgr, yolo_model_path, confidence_threshold=0.5)
                print(f"   YOLO actually used: {used_yolo}, detected {len(bboxes)} person(s)")
                if bboxes and len(bboxes) > 0:
                    bboxes_list = bboxes if process_all_persons else [bboxes[0]]
                    detection_info = f"Detected {len(bboxes)} person(s) by YOLO, processing {len(bboxes_list)}"
                    print(f"✅ {detection_info}")
                else:
                    bboxes_list = [None]  # Process full image
                    detection_info = "No person detected by YOLO, using full image"
                    print(f"⚠️  {detection_info}")
            else:
                bboxes_list = [None]  # Process full image
                detection_info = "YOLO disabled, using full image"
                print(f"⚠️  {detection_info}")
            
            # Step 2-6: Process each person
            # Create black canvas for all pose drawings
            pose_canvas = np.zeros_like(original_image_rgb)
            all_keypoints = []
            all_scores = []
            
            for person_idx, bbox in enumerate(bboxes_list):
                print(f"\n👤 Processing person {person_idx + 1}/{len(bboxes_list)}")
                
                # Step 2: Preprocess image
                print("🔄 Preprocessing image...")
                print(f"   📦 Bbox: {bbox}")
                input_tensor, original_size, crop_info = preprocess_image_for_sdpose(
                    original_image_bgr, bbox, self.input_size
                )
                print(f"   ✂️  Crop info: {crop_info}")
                input_tensor = input_tensor.to(self.device)
                
                # Step 3: Run inference
                print("🚀 Running SDPose inference...")
                test_cfg = {'flip_test': False}
                
                with torch.no_grad():
                    out = self.pipeline(
                        input_tensor,
                        timesteps=[999],
                        test_cfg=test_cfg,
                        show_progress_bar=False,
                        mode="inference",
                    )
                    
                    # Extract keypoints and scores
                    heatmap_inst = out[0]
                    keypoints = heatmap_inst.keypoints[0]  # (K, 2)
                    scores = heatmap_inst.keypoint_scores[0]  # (K,)
                    
                    # Convert to numpy
                    if torch.is_tensor(keypoints):
                        keypoints = keypoints.cpu().numpy()
                    if torch.is_tensor(scores):
                        scores = scores.cpu().numpy()
                
                print(f"📊 Detected {len(keypoints)} keypoints")
                
                # Step 4: Restore coordinates to original space
                if restore_coords and bbox is not None:
                    keypoints_original = restore_keypoints_to_original(
                        keypoints, crop_info, self.input_size, original_size
                    )
                else:
                    scale_x = original_size[0] / self.input_size[0]
                    scale_y = original_size[1] / self.input_size[1]
                    keypoints_original = keypoints.copy()
                    keypoints_original[:, 0] *= scale_x
                    keypoints_original[:, 1] *= scale_y
                
                all_keypoints.append(keypoints_original)
                all_scores.append(scores)
                
                # Step 5: Draw keypoints for this person
                print(f"🎨 Drawing keypoints for person {person_idx + 1}...")
                
                if self.keypoint_scheme == "body":
                    if len(keypoints_original) >= 17:
                        # Draw on pose_canvas (black background, shared by all persons)
                        pose_canvas = draw_body17_keypoints_openpose_style(
                            pose_canvas, keypoints_original[:17], scores[:17], 
                            threshold=score_threshold
                        )
                else:
                    # Wholebody scheme
                    keypoints_with_neck = keypoints_original.copy()
                    scores_with_neck = scores.copy()
                    
                    if len(keypoints_original) >= 17:
                        neck = (keypoints_original[5] + keypoints_original[6]) / 2
                        neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
                        
                        keypoints_with_neck = np.insert(keypoints_original, 17, neck, axis=0)
                        scores_with_neck = np.insert(scores, 17, neck_score)
                        
                        mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                        openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                        
                        temp_kpts = keypoints_with_neck.copy()
                        temp_scores = scores_with_neck.copy()
                        temp_kpts[openpose_idx] = keypoints_with_neck[mmpose_idx]
                        temp_scores[openpose_idx] = scores_with_neck[mmpose_idx]
                        
                        keypoints_with_neck = temp_kpts
                        scores_with_neck = temp_scores
                    
                    # Draw on pose_canvas (black background, shared by all persons)
                    pose_canvas = draw_wholebody_keypoints_openpose_style(
                        pose_canvas, keypoints_with_neck, scores_with_neck, 
                        threshold=score_threshold
                    )
            
            # Blend original image with pose canvas after all persons are drawn
            # overlay_alpha: transparency of (pose + black background) layer
            # 0.0 = invisible (only original image), 1.0 = fully opaque (pose + black bg)
            result_image = cv2.addWeighted(original_image_rgb, 1.0 - overlay_alpha, pose_canvas, overlay_alpha, 0)
            
            # Create info text
            info_text = self._create_info_text(
                original_size, self.input_size, detection_info, bboxes_list[0] if len(bboxes_list) == 1 else None,
                all_keypoints[0] if len(all_keypoints) > 0 else None, 
                all_scores[0] if len(all_scores) > 0 else None, 
                score_threshold,
                len(bboxes_list)
            )
            
            # Generate JSON file
            json_file_path = None
            if all_keypoints and len(all_keypoints) > 0:
                try:
                    # Convert to OpenPose JSON format
                    json_data = convert_to_openpose_json(
                        all_keypoints, all_scores, 
                        original_size[0], original_size[1],
                        self.keypoint_scheme
                    )
                    
                    # Save to temporary file
                    temp_json = tempfile.NamedTemporaryFile(
                        mode='w', suffix='.json', delete=False, 
                        dir=tempfile.gettempdir()
                    )
                    json.dump(json_data, temp_json, indent=2)
                    json_file_path = temp_json.name
                    temp_json.close()
                    
                    print(f"✅ JSON file saved: {json_file_path}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to generate JSON file: {e}")
                    json_file_path = None
            
            print(f"✅ Inference complete. Returning RGB result_image with shape: {result_image.shape}")
            return result_image, all_keypoints, all_scores, info_text, json_file_path
                
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return image, None, None, f"Error during inference: {str(e)}", None
    
    def predict_video(self, video_path, output_path, enable_yolo=True, 
                     yolo_model_path=None, score_threshold=0.3, flip_test=False, overlay_alpha=0.6, progress=gr.Progress()):
        """
        Run inference on a video file
        overlay_alpha: Opacity of pose+black background layer (0.0=invisible, 1.0=fully opaque)
        Returns: (output_video_path, info_text)
        """
        if not self.model_loaded or self.pipeline is None:
            return None, "Model not loaded. Please load the model first."
        
        try:
            # Move models to GPU (only happens inside @spaces.GPU decorated function)
            if self.device.type == 'cuda' and hasattr(self, 'unet_cpu'):
                print("🚀 Moving models to GPU...")
                self.pipeline.unet = self.unet_cpu.to(self.device)
                self.pipeline.vae = self.vae_cpu.to(self.device)
                self.pipeline.text_encoder = self.text_encoder_cpu.to(self.device)
                self.pipeline.decoder = self.hm_decoder_cpu.to(self.device)
                print("✅ Models on GPU")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, f"Error: Could not open video {video_path}"
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # Default fallback
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Processing video: {total_frames} frames at {fps} FPS, size {width}x{height}")
            
            # Create video writer
            # Use mp4v for initial encoding (will re-encode to H.264 later if needed)
            print(f"📝 Creating VideoWriter with mp4v codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Ensure output path has .mp4 extension
            actual_output_path = output_path
            if not actual_output_path.endswith('.mp4'):
                actual_output_path = output_path.rsplit('.', 1)[0] + '.mp4'
            
            out = cv2.VideoWriter(actual_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                print(f"❌ Failed to open VideoWriter")
                return None, f"Error: Could not create video writer"
            
            print(f"✅ VideoWriter opened successfully: {actual_output_path}")
            
            frame_count = 0
            processed_count = 0
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress
                if progress is not None:
                    progress((frame_count, total_frames), desc=f"Processing frame {frame_count}/{total_frames}")
                
                # Convert frame from BGR to RGB for predict_image
                # cv2.VideoCapture reads in BGR format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference on frame (frame_rgb is RGB)
                # Process all detected persons
                result_frame, _, _, _, _ = self.predict_image(
                    frame_rgb, enable_yolo=enable_yolo, yolo_model_path=yolo_model_path,
                    score_threshold=score_threshold, restore_coords=True, flip_test=flip_test, 
                    process_all_persons=True, overlay_alpha=overlay_alpha
                )
                
                if result_frame is not None:
                    # result_frame is RGB from predict_image, convert to BGR for video writing
                    result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    
                    # Check frame size matches
                    if result_frame_bgr.shape[:2] != (height, width):
                        print(f"⚠️  Frame size mismatch: {result_frame_bgr.shape[:2]} vs expected ({height}, {width}), resizing...")
                        result_frame_bgr = cv2.resize(result_frame_bgr, (width, height))
                    
                    out.write(result_frame_bgr)
                    processed_count += 1
                else:
                    # If inference failed, write original frame (already BGR)
                    print(f"⚠️  Frame {frame_count} inference failed, using original")
                    out.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames, written {processed_count}")
            
            cap.release()
            out.release()
            
            # Ensure the video file is properly written and flushed
            # Small delay to ensure file system has finished writing
            import time
            time.sleep(0.5)
            
            # Verify the output file exists and has content
            if not os.path.exists(actual_output_path):
                return None, f"Error: Output video file was not created at {actual_output_path}"
            
            file_size = os.path.getsize(actual_output_path)
            if file_size == 0:
                return None, f"Error: Output video file is empty (0 bytes)"
            
            print(f"✅ Video file created: {actual_output_path} ({file_size} bytes)")
            
            # If we used mp4v codec, try to re-encode to H.264 for better browser compatibility
            final_output_path = actual_output_path
            if actual_output_path.endswith('.mp4'):
                try:
                    import subprocess
                    print("🔄 Re-encoding video to H.264 for better browser compatibility...")
                    
                    # Create a new temp file for H.264 version
                    h264_path = actual_output_path.rsplit('.', 1)[0] + '_h264.mp4'
                    
                    # Use ffmpeg to re-encode
                    cmd = [
                        'ffmpeg', '-y', '-i', actual_output_path,
                        '-c:v', 'libx264', '-preset', 'fast', 
                        '-crf', '23', '-pix_fmt', 'yuv420p',
                        h264_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(h264_path):
                        h264_size = os.path.getsize(h264_path)
                        if h264_size > 0:
                            print(f"✅ Re-encoded to H.264: {h264_path} ({h264_size} bytes)")
                            # Use the H.264 version
                            final_output_path = h264_path
                            file_size = h264_size
                            # Remove the original mp4v version
                            try:
                                os.unlink(actual_output_path)
                            except:
                                pass
                        else:
                            print(f"⚠️  Re-encoded file is empty, using original")
                    else:
                        print(f"⚠️  Re-encoding failed, using original mp4v version")
                        if result.stderr:
                            print(f"   ffmpeg error: {result.stderr.decode()[:200]}")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Re-encoding timed out, using original")
                except Exception as e:
                    print(f"⚠️  Re-encoding failed: {e}, using original")
            
            info_text = f"✅ Video processing complete!\n"
            info_text += f"📊 Total frames: {total_frames}\n"
            info_text += f"✓ Processed: {processed_count}\n"
            info_text += f"🎞️ FPS: {fps}\n"
            info_text += f"📏 Resolution: {width}x{height}\n"
            info_text += f"💾 File size: {file_size / (1024*1024):.2f} MB\n"
            info_text += f"💾 Output saved to: {final_output_path}"
            
            print(info_text)
            return final_output_path, info_text
            
        except Exception as e:
            print(f"Error during video inference: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error during video inference: {str(e)}"
    
    def _create_info_text(self, original_size, input_size, detection_info, bbox,
                         keypoints, scores, threshold, num_persons=1):
        """Create informative text about the inference results"""
        info_text = "🎯 SDPose Keypoint Detection Results\n" + "="*60 + "\n"
        info_text += f"📏 Original Image Size: {original_size}\n"
        info_text += f"🔧 Model Input Size: {input_size}\n"
        info_text += f"🧠 Keypoint Scheme: {self.keypoint_scheme}\n"
        info_text += f"🔍 Detection: {detection_info}\n"
        info_text += f"👥 Number of Persons Processed: {num_persons}\n"
        if bbox:
            info_text += f"📦 Bounding Box (first person): [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]\n"
        info_text += f"🎚️ Score Threshold: {threshold}\n"
        info_text += "="*60 + "\n\n"
        
        # Count detected keypoints (for first person if available)
        if keypoints is not None and scores is not None:
            detected_count = np.sum(scores >= threshold)
            total_count = len(scores)
            info_text += f"📊 Summary (first person): {detected_count}/{total_count} keypoints detected above threshold\n"
        
        info_text += f"🎨 Visualization: Openpose style\n"
        info_text += f"📍 Coordinates: Restored to original image space\n"
        
        return info_text


# Global instances for both models
inference_engines = {
    "body": SDPoseInference(),
    "wholebody": SDPoseInference()
}


def switch_model(model_type):
    """Switch between models"""
    if not inference_engines[model_type].model_loaded:
        print(f"🔄 Loading {model_type} model...")
        success = inference_engines[model_type].load_model_from_hub(keypoint_scheme=model_type)
        if success:
            return f"✅ {model_type.capitalize()} model loaded!"
        else:
            return f"❌ Failed to load {model_type} model"
    else:
        return f"✅ {model_type.capitalize()} model ready"


@spaces.GPU(duration=120)
def run_inference_image(image, model_type, enable_yolo, score_threshold, overlay_alpha):
    """Image inference interface with zero_gpu support"""
    if image is None:
        return None, None, "Please upload an image"
    
    if not inference_engines[model_type].model_loaded:
        status = switch_model(model_type)
        if "Failed" in status:
            return image, None, status
    
    result_image, _, _, info_text, json_file = inference_engines[model_type].predict_image(
        image, enable_yolo=enable_yolo,
        score_threshold=score_threshold, overlay_alpha=overlay_alpha
    )
    
    return result_image, json_file, info_text


@spaces.GPU(duration=600)
def run_inference_video(video, model_type, enable_yolo, score_threshold, overlay_alpha, progress=gr.Progress()):
    """Video inference interface with zero_gpu support"""
    if video is None:
        return None, None, "Please upload a video"
    
    if not inference_engines[model_type].model_loaded:
        status = switch_model(model_type)
        if "Failed" in status:
            return None, None, status
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_file.name
    temp_file.close()
    
    result_video, info_text = inference_engines[model_type].predict_video(
        video, output_path, enable_yolo=enable_yolo,
        score_threshold=score_threshold, overlay_alpha=overlay_alpha,
        progress=progress
    )
    
    if result_video and os.path.exists(result_video):
        return result_video, result_video, info_text
    else:
        return None, None, info_text


def create_gradio_interface():
    """Create Gradio interface"""

    logo_path = "assets/logo/logo.png"
    
    with gr.Blocks(title="SDPose - Gradio Interface", theme=gr.themes.Soft()) as demo:
        
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=1, min_width=150):
                gr.Image(value=str(logo_path), show_label=False, show_download_button=False, 
                        show_share_button=False, container=False, height=150, width=150, 
                        interactive=False, show_fullscreen_button=False)
            
            with gr.Column(scale=9):
                gr.HTML("""
                <div style="text-align: left; padding: 10px;">
                    <h1 style="margin-bottom: 20px; font-size: 2.2em; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                        SDPose: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation
                    </h1>
                    <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 15px;">
                        <a href="https://arxiv.org/abs/2509.24980" target="_blank" 
                           style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; border-radius: 8px; font-weight: 600; text-decoration: none !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); transition: all 0.3s ease; cursor: pointer;"
                           onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(102, 126, 234, 0.5)';"
                           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(102, 126, 234, 0.4)';">
                            📄 Paper
                        </a>
                        <a href="https://github.com/T-S-Liang/SDPose-OOD" target="_blank" 
                           style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white !important; border-radius: 8px; font-weight: 600; text-decoration: none !important; box-shadow: 0 4px 12px rgba(245, 87, 108, 0.4); transition: all 0.3s ease; cursor: pointer;"
                           onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(245, 87, 108, 0.5)';"
                           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(245, 87, 108, 0.4)';">
                            💻 GitHub
                        </a>
                        <a href="https://huggingface.co/teemosliang/SDPose-Body" target="_blank" 
                           style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white !important; border-radius: 8px; font-weight: 600; text-decoration: none !important; box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4); transition: all 0.3s ease; cursor: pointer;"
                           onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(79, 172, 254, 0.5)';"
                           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(79, 172, 254, 0.4)';">
                            🤗 Body Model
                        </a>
                        <a href="https://huggingface.co/teemosliang/SDPose-Wholebody" target="_blank" 
                           style="display: inline-block; padding: 10px 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white !important; border-radius: 8px; font-weight: 600; text-decoration: none !important; box-shadow: 0 4px 12px rgba(67, 233, 123, 0.4); transition: all 0.3s ease; cursor: pointer;"
                           onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(67, 233, 123, 0.5)';"
                           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(67, 233, 123, 0.4)';">
                            🤗 WholeBody Model
                        </a>
                    </div>
                </div>
                """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Settings")
                
                model_type = gr.Radio(
                    choices=["body", "wholebody"],
                    value="body",
                    label="Model Selection",
                    info="Body (17 kpts) or WholeBody (133 kpts)"
                )
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Select model and upload media",
                    interactive=False
                )
                
                enable_yolo = gr.Checkbox(
                    label="Enable YOLO Detection",
                    value=True,
                    info="For multi-person detection"
                )
                
                score_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="Confidence Threshold"
                )
                
                overlay_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                    label="Pose Overlay Opacity"
                )
            
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("📷 Image"):
                        with gr.Row():
                            input_image = gr.Image(label="Input Image", type="numpy", height=400)
                            output_image = gr.Image(label="Output with Keypoints", height=400)
                        
                        with gr.Row():
                            output_json = gr.File(label="📥 Download JSON", scale=1)
                            image_info = gr.Textbox(label="Detection Results", lines=6, max_lines=10, scale=1)
                        
                        run_image_btn = gr.Button("🔍 Run Image Inference", variant="primary", size="lg")
                    
                    with gr.Tab("🎬 Video"):
                        with gr.Row():
                            input_video = gr.Video(label="Input Video", height=400)
                            output_video = gr.Video(label="Output Video with Keypoints", height=400)
                        
                        with gr.Row():
                            output_video_file = gr.File(label="📥 Download Processed Video", scale=1)
                            video_info = gr.Textbox(label="Processing Results", lines=6, max_lines=10, scale=1)
                        
                        run_video_btn = gr.Button("🎬 Run Video Inference", variant="primary", size="lg")
        
        gr.Markdown("""
        ### 📝 Usage
        1. Select model (Body or WholeBody)
        2. Upload image or video
        3. Configure settings
        4. Click Run button
        5. Download results
        
        ### ⚠️ Notes
        - First load may take 1-2 minutes
        - YOLO-det recommended for multi-person
        - Video processing may be slow on CPU
        """)
        
        # Events
        model_type.change(
            fn=switch_model,
            inputs=[model_type],
            outputs=[model_status]
        )
        
        run_image_btn.click(
            fn=run_inference_image,
            inputs=[input_image, model_type, enable_yolo, score_threshold, overlay_alpha],
            outputs=[output_image, output_json, image_info]
        )
        
        run_video_btn.click(
            fn=run_inference_video,
            inputs=[input_video, model_type, enable_yolo, score_threshold, overlay_alpha],
            outputs=[output_video, output_video_file, video_info]
        )
    
    return demo


# Pre-load body model
print("=" * 60)
print("🚀 SDPose Space Starting...")
print("=" * 60)
if SPACES_ZERO_GPU:
    print("✅ zero_gpu enabled")
else:
    print("⚠️  zero_gpu disabled (running on standard hardware)")

print("🔄 Pre-loading Body model...")
success = inference_engines["body"].load_model_from_hub(keypoint_scheme="body")
if success:
    print("✅ Body model ready!")
else:
    print("⚠️  Body model will load on demand")

print("ℹ️  WholeBody model will load when selected")
print("=" * 60)

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
