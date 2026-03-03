import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json

signal = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
output_path = "/home/gsantm/scripts/pose_estimators/mediapipe_holsitic/output"
model_path = os.path.expanduser("~/store/pose_estimators/mediapipe_holistic/pose_landmarker_lite.task")

os.makedirs(output_path, exist_ok=True)

# Initialize mediapipe base options and landmarker options
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the pose landmarker in VIDEO mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(signal)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {signal}")

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {frame_count}")

    frame_idx = 0
    results_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Compute timestamp in milliseconds
        timestamp_ms = int((frame_idx / fps) * 1000)

        # Perform pose landmarking
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extract landmarks
        if result.pose_landmarks:
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in result.pose_landmarks[0]
            ]
        else:
            landmarks = []

        results_list.append({
            "frame": frame_idx,
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks
        })

        frame_idx += 1

    cap.release()

# Save results to JSON
output_json = os.path.join(output_path, os.path.basename(signal).replace(".mp4", "_pose.json"))
with open(output_json, "w") as f:
    json.dump(results_list, f, indent=2)

print(f"\n✅ Pose estimation complete. Results saved to: {output_json}")
