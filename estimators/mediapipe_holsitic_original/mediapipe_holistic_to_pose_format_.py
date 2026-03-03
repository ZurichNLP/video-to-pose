from pose_format.utils.holistic import load_mediapipe_directory
from pose_format.utils.openpose import hand_colors, load_frames_directory_dict

signal = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
output_path = "/home/gsantm/store/pose_estimators/mediapipe_holistic/output/-fZc293MpJk_0-1-rgb_front"

fps=50
width=674
height=588

pose = load_mediapipe_directory(output_path, fps=fps, width=width, height=height)

with open(f"{output_path}/pose.pose", "wb") as data_buffer:
    pose.write(data_buffer)