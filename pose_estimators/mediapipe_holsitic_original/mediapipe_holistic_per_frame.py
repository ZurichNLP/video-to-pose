import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json

# -----------------------------
# User parameters
# -----------------------------
signal = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
output_root = "/home/gsantm/store/pose_estimators/mediapipe_holistic/output"
model_path = os.path.expanduser("~/store/pose_estimators/mediapipe_holistic/pose_landmarker_lite.task")

# -----------------------------
# Prepare output directories
# -----------------------------
sample_name = os.path.splitext(os.path.basename(signal))[0]
output_dir = os.path.join(output_root, sample_name)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Mediapipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# -----------------------------
# Process video frame-by-frame
# -----------------------------
print(options)
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(signal)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {signal}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {frame_count}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run per-frame pose estimation
        result = landmarker.detect(mp_image)
        print(f"result: {result}")

        # Extract pose landmarks if available
        if result.pose_landmarks:
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in result.pose_landmarks[0]
            ]
        else:
            landmarks = []

        frame_json_path = os.path.join(output_dir, f"frame_{frame_idx}.json")
        with open(frame_json_path, "w") as f:
            json.dump({
                "frame": frame_idx,
                "landmarks": landmarks
            }, f, indent=2)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames...")

    cap.release()

print(f"\n✅ Done! Saved {frame_idx} JSON files to:")
print(f"   {output_dir}")