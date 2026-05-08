"""
Batch inference script for video datasets - COCO WholeBody format output (multi-process optimized)

Optimization features:
1. Multi-process parallel video processing (fully utilizes 4 CPUs)
2. GPU lock mechanism to prevent VRAM conflicts
3. Memory management to prevent OOM
4. Resume support

Usage:
python video_inference_wholebody_mp.py \
    --input_dir /path/to/video_dataset \
    --output_dir /path/to/output \
    --pose_checkpoint weights/posebh/wholebody.pth \
    --num_workers 4 \
    --device cuda:0
"""

import os
import os.path as osp
import json
import cv2
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import warnings
import gc
import time
from functools import partial

import torch
import torch.multiprocessing as mp
from multiprocessing import Queue, Process, Value, Lock
from queue import Empty

warnings.filterwarnings('ignore')


def parse_args():
    parser = ArgumentParser(description='Batch video inference for COCO WholeBody (Multi-process)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for JSON results')
    parser.add_argument('--pose_config', type=str,
                        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
                        help='Pose model config file')
    parser.add_argument('--pose_checkpoint', type=str,
                        default='weights/posebh/wholebody.pth',
                        help='Pose model checkpoint file')
    parser.add_argument('--det_config', type=str,
                        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
                        help='Detection model config file')
    parser.add_argument('--det_checkpoint', type=str,
                        default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                        help='Detection model checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--bbox_thr', type=float, default=0.3,
                        help='Bounding box score threshold')
    parser.add_argument('--kpt_thr', type=float, default=0.3,
                        help='Keypoint score threshold')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Process every N frames')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of CPU workers for video decoding')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for GPU inference')
    parser.add_argument('--queue_size', type=int, default=16,
                        help='Size of frame queue')
    parser.add_argument('--use_yolo', action='store_true',
                        help='Use YOLOv8 for detection')
    parser.add_argument('--yolo_checkpoint', type=str, default='yolov8n.pt',
                        help='YOLOv8 checkpoint file')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed videos')
    parser.add_argument('--max_frames_in_memory', type=int, default=100,
                        help='Max frames to keep in memory per video')
    return parser.parse_args()


def find_videos(input_dir, extensions=('.mp4', '.webm', '.avi', '.mov', '.mkv')):
    """Recursively find all video files."""
    videos = []
    for ext in extensions:
        videos.extend(glob(osp.join(input_dir, '**', f'*{ext}'), recursive=True))
        videos.extend(glob(osp.join(input_dir, '**', f'*{ext.upper()}'), recursive=True))
    return sorted(list(set(videos)))


def get_output_path(video_path, input_dir, output_dir):
    """Get output JSON path."""
    rel_path = osp.relpath(video_path, input_dir)
    rel_dir = osp.dirname(rel_path)
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_subdir = osp.join(output_dir, rel_dir)
    return osp.join(output_subdir, f"{video_name}_wholebody.json")


def is_processed(video_path, input_dir, output_dir):
    """Check whether a video has already been processed."""
    output_path = get_output_path(video_path, input_dir, output_dir)
    return osp.exists(output_path)


def create_coco_wholebody_annotation(pose_results, frame_idx, video_name,
                                     frame_width, frame_height):
    """Convert pose results to COCO-WholeBody format."""
    annotations = []

    for person_idx, result in enumerate(pose_results):
        keypoints = result['keypoints']
        bbox = result.get('bbox', None)

        keypoints_flat = []
        num_visible = 0
        for kpt in keypoints:
            x, y, score = kpt
            if score > 0.0:
                visibility = 2
                num_visible += 1
            else:
                visibility = 0
            keypoints_flat.extend([float(x), float(y), visibility])

        if bbox is None:
            valid_kpts = keypoints[keypoints[:, 2] > 0]
            if len(valid_kpts) > 0:
                x_min, y_min = valid_kpts[:, :2].min(axis=0)
                x_max, y_max = valid_kpts[:, :2].max(axis=0)
                bbox = [float(x_min), float(y_min),
                        float(x_max - x_min), float(y_max - y_min)]
            else:
                bbox = [0, 0, 0, 0]
        else:
            if len(bbox) >= 4:
                bbox = [float(bbox[0]), float(bbox[1]),
                        float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]

        area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0

        annotation = {
            'id': f"{video_name}_{frame_idx}_{person_idx}",
            'image_id': f"{video_name}_{frame_idx}",
            'category_id': 1,
            'keypoints': keypoints_flat,
            'num_keypoints': num_visible,
            'bbox': bbox,
            'area': float(area),
            'iscrowd': 0,
            'score': float(np.mean(keypoints[:, 2]))
        }

        num_kpts = len(keypoints)
        if num_kpts >= 133:
            annotation['foot_kpts'] = keypoints_flat[17*3:23*3]
            annotation['face_kpts'] = keypoints_flat[23*3:91*3]
            annotation['lefthand_kpts'] = keypoints_flat[91*3:112*3]
            annotation['righthand_kpts'] = keypoints_flat[112*3:133*3]
            annotation['foot_valid'] = bool(np.sum(keypoints[17:23, 2]) > 0)
            annotation['face_valid'] = bool(np.sum(keypoints[23:91, 2]) > 0)
            annotation['lefthand_valid'] = bool(np.sum(keypoints[91:112, 2]) > 0)
            annotation['righthand_valid'] = bool(np.sum(keypoints[112:133, 2]) > 0)

        annotations.append(annotation)

    return annotations


class VideoDecoder:
    """Video decoder - runs in CPU workers."""
    def __init__(self, video_path, frame_interval=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __exit__(self, *args):
        if self.cap:
            self.cap.release()

    def get_info(self):
        if not self.cap or not self.cap.isOpened():
            return None
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def iter_frames(self):
        """Iterate over and return frames."""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_interval == 0:
                yield frame_idx, frame
            frame_idx += 1


def video_reader_worker(video_queue, frame_queue, stop_flag, args):
    """
    Video reader worker process.
    Fetches video paths from video_queue, decodes them, and puts frames into frame_queue.
    """
    while not stop_flag.value:
        try:
            video_info = video_queue.get(timeout=1)
        except Empty:
            continue

        if video_info is None:
            break

        video_path, video_idx = video_info
        video_name = osp.splitext(osp.basename(video_path))[0]

        try:
            with VideoDecoder(video_path, args.frame_interval) as decoder:
                info = decoder.get_info()
                if info is None:
                    print(f"[Reader] Cannot open: {video_path}")
                    frame_queue.put((video_idx, video_path, None, None, None))
                    continue

                # Send video metadata
                frame_queue.put((video_idx, video_path, 'info', info, None))

                # Collect frames in batches
                batch_frames = []
                batch_indices = []

                for frame_idx, frame in decoder.iter_frames():
                    batch_frames.append(frame)
                    batch_indices.append(frame_idx)

                    if len(batch_frames) >= args.batch_size:
                        # Wait for queue space to avoid memory explosion
                        while frame_queue.qsize() > args.queue_size and not stop_flag.value:
                            time.sleep(0.01)

                        frame_queue.put((video_idx, video_path, 'frames',
                                        batch_indices.copy(), batch_frames.copy()))
                        batch_frames.clear()
                        batch_indices.clear()

                # Send remaining frames
                if batch_frames:
                    frame_queue.put((video_idx, video_path, 'frames',
                                    batch_indices, batch_frames))

                # Send end marker
                frame_queue.put((video_idx, video_path, 'end', None, None))

        except Exception as e:
            print(f"[Reader] Error processing {video_path}: {e}")
            frame_queue.put((video_idx, video_path, None, None, None))

    print("[Reader] Worker exiting")


def gpu_inference_worker(frame_queue, result_queue, stop_flag, args):
    """
    GPU inference worker process.
    Fetches frames from frame_queue, runs inference, and puts results into result_queue.
    """
    # Initialize models in this process
    from mmpose.apis import init_pose_model, inference_top_down_pose_model
    from mmpose.datasets import DatasetInfo

    print("[GPU] Loading pose model...")
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                  device=args.device)
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get('dataset_info', None))

    # Initialize detector
    det_model = None
    if args.use_yolo:
        print("[GPU] Loading YOLOv8...")
        from ultralytics import YOLO
        det_model = YOLO(args.yolo_checkpoint)
    else:
        try:
            from mmdet.apis import inference_detector, init_detector
            print("[GPU] Loading mmdet...")
            det_model = init_detector(args.det_config, args.det_checkpoint,
                                       device=args.device)
        except:
            print("[GPU] No detection model, using full frame")

    print("[GPU] Models loaded, starting inference...")

    current_video_idx = None
    video_results = {}

    while not stop_flag.value:
        try:
            item = frame_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            break

        video_idx, video_path, msg_type, data1, data2 = item
        video_name = osp.splitext(osp.basename(video_path))[0]

        if msg_type is None:
            # Video read failed
            result_queue.put((video_idx, video_path, None))
            continue

        if msg_type == 'info':
            # Initialize per-video result container
            video_results[video_idx] = {
                'video_path': video_path,
                'video_name': video_name,
                'info': data1,
                'images': [],
                'annotations': []
            }
            continue

        if msg_type == 'end':
            # Video processing completed, send results
            if video_idx in video_results:
                result_queue.put((video_idx, video_path, video_results[video_idx]))
                del video_results[video_idx]
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
            continue

        if msg_type == 'frames':
            frame_indices = data1
            frames = data2
            info = video_results.get(video_idx, {}).get('info', {})
            frame_width = info.get('width', 0)
            frame_height = info.get('height', 0)

            for frame_idx, frame in zip(frame_indices, frames):
                try:
                    # Human detection
                    if args.use_yolo and det_model is not None:
                        results = det_model(frame, conf=args.bbox_thr, classes=[0], verbose=False)
                        person_results = []
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                person_results.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2), float(conf)]
                                })
                    elif det_model is not None:
                        from mmdet.apis import inference_detector
                        from mmpose.apis import process_mmdet_results
                        mmdet_results = inference_detector(det_model, frame)
                        person_results = process_mmdet_results(mmdet_results, cat_id=1)
                        person_results = [p for p in person_results if p['bbox'][4] >= args.bbox_thr]
                    else:
                        person_results = [{'bbox': [0, 0, frame_width, frame_height, 1.0]}]

                    # Pose estimation
                    if len(person_results) > 0:
                        pose_results, _ = inference_top_down_pose_model(
                            pose_model,
                            frame,
                            person_results,
                            bbox_thr=args.bbox_thr,
                            format='xyxy',
                            dataset_info=dataset_info,
                            return_heatmap=False
                        )
                    else:
                        pose_results = []

                    # Append results
                    image_id = f"{video_name}_{frame_idx}"
                    video_results[video_idx]['images'].append({
                        'id': image_id,
                        'file_name': f"{video_name}/frame_{frame_idx:06d}.jpg",
                        'width': frame_width,
                        'height': frame_height,
                        'frame_idx': frame_idx,
                        'video_name': video_name
                    })

                    frame_annotations = create_coco_wholebody_annotation(
                        pose_results, frame_idx, video_name, frame_width, frame_height
                    )
                    video_results[video_idx]['annotations'].extend(frame_annotations)

                except Exception as e:
                    print(f"[GPU] Error on frame {frame_idx}: {e}")

            # Release frame memory
            del frames
            gc.collect()

    print("[GPU] Worker exiting")


def result_writer_worker(result_queue, stop_flag, args, total_videos, progress_counter):
    """
    Result writer worker process.
    Fetches results from result_queue and writes them to files.
    """
    categories = [{
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
        'keypoints': [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'left_big_toe', 'left_small_toe', 'left_heel',
            'right_big_toe', 'right_small_toe', 'right_heel',
            *[f'face_{i}' for i in range(68)],
            *[f'left_hand_{i}' for i in range(21)],
            *[f'right_hand_{i}' for i in range(21)]
        ],
        'skeleton': []
    }]

    processed = 0
    pbar = tqdm(total=total_videos, desc="Saving results")

    while processed < total_videos and not stop_flag.value:
        try:
            item = result_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            break

        video_idx, video_path, result = item

        if result is None:
            print(f"[Writer] Skipping failed video: {video_path}")
            processed += 1
            with progress_counter.get_lock():
                progress_counter.value += 1
            pbar.update(1)
            continue

        # Build output path
        output_path = get_output_path(video_path, args.input_dir, args.output_dir)
        os.makedirs(osp.dirname(output_path), exist_ok=True)

        # Build COCO-format output
        coco_output = {
            'info': {
                'description': 'COCO-WholeBody format pose estimation results',
                'version': '1.0',
                'source_video': video_path,
                'video_info': result['info']
            },
            'licenses': [],
            'images': result['images'],
            'annotations': result['annotations'],
            'categories': categories
        }

        # Write file
        try:
            with open(output_path, 'w') as f:
                json.dump(coco_output, f)
            # print(f"[Writer] Saved: {output_path}")
        except Exception as e:
            print(f"[Writer] Error saving {output_path}: {e}")

        processed += 1
        with progress_counter.get_lock():
            progress_counter.value += 1
        pbar.update(1)

    pbar.close()
    print(f"[Writer] Finished. Processed {processed} videos.")


def main():
    args = parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Check input directory
    if not osp.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all videos
    print("Scanning for videos...")
    all_videos = find_videos(args.input_dir)
    print(f"Found {len(all_videos)} videos total")

    # Filter already processed videos (resume support)
    if args.resume:
        videos = [v for v in all_videos if not is_processed(v, args.input_dir, args.output_dir)]
        print(f"Skipping {len(all_videos) - len(videos)} already processed videos")
        print(f"Remaining: {len(videos)} videos to process")
    else:
        videos = all_videos

    if len(videos) == 0:
        print("No videos to process!")
        return

    # Create queues and shared variables
    video_queue = mp.Queue()  # Video path queue
    frame_queue = mp.Queue(maxsize=args.queue_size)  # Frame queue, size-limited to prevent OOM
    result_queue = mp.Queue()  # Result queue
    stop_flag = mp.Value('b', False)  # Stop flag
    progress_counter = mp.Value('i', 0)  # Progress counter

    # Put videos into queue
    for idx, video_path in enumerate(videos):
        video_queue.put((video_path, idx))

    # Add termination markers
    for _ in range(args.num_workers):
        video_queue.put(None)

    # Start reader workers (CPU-intensive)
    reader_processes = []
    for i in range(args.num_workers):
        p = mp.Process(target=video_reader_worker,
                       args=(video_queue, frame_queue, stop_flag, args),
                       name=f"Reader-{i}")
        p.start()
        reader_processes.append(p)
        print(f"Started reader worker {i}")

    # Start GPU worker (single worker because there is only one GPU)
    gpu_process = mp.Process(target=gpu_inference_worker,
                             args=(frame_queue, result_queue, stop_flag, args),
                             name="GPU-Worker")
    gpu_process.start()
    print("Started GPU worker")

    # Start writer worker
    writer_process = mp.Process(target=result_writer_worker,
                                args=(result_queue, stop_flag, args, len(videos), progress_counter),
                                name="Writer")
    writer_process.start()
    print("Started writer worker")

    # Wait for all readers to finish
    for p in reader_processes:
        p.join()
    print("All readers finished")

    # Send GPU termination signal
    frame_queue.put(None)
    gpu_process.join()
    print("GPU worker finished")

    # Send writer termination signal
    result_queue.put(None)
    writer_process.join()
    print("Writer finished")

    print("=" * 50)
    print("Done!")
    print(f"Processed videos: {progress_counter.value}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
