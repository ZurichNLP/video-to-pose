import cv2
import os
import sys
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from .base_estimator import BasePoseEstimator
import imageio

base_dir = os.getcwd()
data_dir = f"{base_dir}/data" 
visualization_dir = f"{base_dir}/vis"
model_path = f"{base_dir}/models/hand_landmarker.task"

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def estimate(video_path, frames):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO
    )
    
    with HandLandmarker.create_from_options(options) as detector:
        timestamp = 0
        detection_results = []
        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect_for_video(mp_image, timestamp)
            detection_results.append(detection_result)
            timestamp += 33
        return detection_results
    
def visualize(video_name, frames, poses):
    annotated_frames = []

    output_path = f"{visualization_dir}/{video_name}"

    for frame, detection in zip(frames, poses):
        annotated_frame = draw_landmarks(frame, detection)
        annotated_frames.append(annotated_frame)

    writer = imageio.get_writer(output_path, fps=30)
    for f in annotated_frames:
        writer.append_data(f)  
    writer.close()

def draw_landmarks(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def load_video_frames(path_to_video):
    frames = []
    
    try:
        reader = imageio.get_reader(path_to_video)
    except Exception as e:
        raise ValueError(f"Could not open video file: {path_to_video}\n{e}")

    for frame in reader:
        frames.append(frame)
    
    reader.close()
    return frames

def main():
    print(f"Beginning pose estimation with mediapipehands.")


    for video_name in os.listdir(data_dir):
        print("\n\nAttempting to estimate pose for video: " + video_name) 
        video_path = os.path.join(data_dir, video_name) # get full path as a string 

        frames = load_video_frames(video_path) 
        poses = estimate(video_path, frames)
        visualize(video_name, frames, poses) # output saved to /vis directory
        print(f"Estimation and visualization for video {video_name} is complete.\n\n")

    print(f"Pose estimation with mediapipehands is complete.")
    
if __name__ == "__main__":
    main()