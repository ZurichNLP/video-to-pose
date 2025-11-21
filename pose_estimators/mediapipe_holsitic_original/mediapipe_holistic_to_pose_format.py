
from pose_format.utils.holistic import load_mediapipe_directory
from pose_format.utils.openpose import hand_colors, load_frames_directory_dict

signal = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
output_path = "/home/gsantm/store/pose_estimators/mediapipe_holistic/output/-fZc293MpJk_0-1-rgb_front"

fps=50
width=674
height=588

# frames = load_frames_directory_dict(directory=output_path, pattern="(?:^|\D)?(\d+).*?.json")

# print(f"frames: {frames}")

pose = load_mediapipe_directory(output_path, fps=fps, width=width, height=height)

with open(f"{output_path}/pose.pose", "wb") as data_buffer:
    pose.write(data_buffer)


# from pose_format.utils.holistic import load_mediapipe_directory
# from pose_format.utils.openpose import hand_colors, load_frames_directory_dict
# import os
# import json

# signal = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
# output_path = "/home/gsantm/store/pose_estimators/mediapipe_holistic/output/-fZc293MpJk_0-1-rgb_front"

# fps = 50
# width = 674
# height = 588

# frames = load_frames_directory_dict(directory=output_path, pattern="(?:^|\\D)?(\\d+).*?.json")

# print(f"✅ Loaded {len(frames)} frames")

# # Convert NumPy or other non-serializable objects to lists if necessary
# def make_json_serializable(obj):
#     if isinstance(obj, (list, dict)):
#         return obj
#     elif hasattr(obj, "tolist"):
#         return obj.tolist()
#     else:
#         return str(obj)

# # Save the entire structure to a JSON file
# frames_json_path = os.path.join(output_path, "frames_structure.json")

# with open(frames_json_path, "w") as f:
#     json.dump(frames, f, indent=2, default=make_json_serializable)

# print(f"💾 Saved frames structure to {frames_json_path}")
# print(f"👉 You can now open it with VS Code or jq to inspect the full structure.")
