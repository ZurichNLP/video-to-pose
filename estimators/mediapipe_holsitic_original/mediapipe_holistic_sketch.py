#!/usr/bin/env python
import argparse
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES

import cv2
from pose_format.utils.holistic import load_holistic

mp_holistic = mp.solutions.holistic

def load_video_frames(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

# def pose_video(input_path: str, output_path: str, format: str, additional_config: dict = {'model_complexity': 1}, progress: bool = True):
#     # Load video frames
#     print('Loading video ...')
#     cap = cv2.VideoCapture(input_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = load_video_frames(cap)

#     # Perform pose estimation
#     print('Estimating pose ...')
#     if format == 'mediapipe':
#         pose = load_holistic(frames,
#                              fps=fps,
#                              width=width,
#                              height=height,
#                              progress=progress,
#                              additional_holistic_config=additional_config)
#     else:
#         raise NotImplementedError('Pose format not supported')

#     # Write
#     print('Saving to disk ...')
#     with open(output_path, "wb") as f:
#         pose.write(f)

def process_holistic(frames: list,
                     fps: float,
                     w: int,
                     h: int,
                     kinect=None,
                     progress=False,
                     additional_face_points=0,
                     additional_holistic_config={}):

    if 'static_image_mode' not in additional_holistic_config:
        additional_holistic_config['static_image_mode'] = False
    holistic = mp_holistic.Holistic(**additional_holistic_config)

    try:
        datas = []
        confs = []

        for i, frame in enumerate(tqdm(frames, disable=not progress)):
            results = holistic.process(frame)
            print(f"results[{i}]: {results}")

            # body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33)
            # face_data, face_confidence = component_points(results.face_landmarks, w, h,
            #                                               FACE_POINTS_NUM(additional_face_points))
            # lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21)
            # rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)
            # body_world_data, body_world_confidence = body_points(results.pose_world_landmarks, w, h, 33)

            # data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data])
            # conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence])

            # if kinect is not None:
            #     kinect_depth = []
            #     for x, y, z in np.array(data, dtype="int32"):
            #         if 0 < x < w and 0 < y < h:
            #             kinect_depth.append(kinect[i, y, x, 0])
            #         else:
            #             kinect_depth.append(0)

            #     kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
            #     data = np.concatenate([data, kinect_vec], axis=-1)

            # datas.append(data)
            # confs.append(conf)

        # pose_body_data = np.expand_dims(np.stack(datas), axis=1)
        # pose_body_conf = np.expand_dims(np.stack(confs), axis=1)

        # return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)
    finally:
        holistic.close()
    return results



input_path = "/home/gsantm/store/data/aligned_yolo_cropped_how2sign/test/clips/-fZc293MpJk_0-1-rgb_front.mp4"
output_path = "/home/gsantm/scripts/pose_estimators/mediapipe_holsitic/output"
model_path = os.path.expanduser("~/store/pose_estimators/mediapipe_holistic/pose_landmarker_heavy.task")


cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = load_video_frames(cap)

results=process_holistic(frames=frames, fps=fps, w=width, h=height)


def process_holistic(frames: list, fps: float, w: int, h: int, kinect=None, progress=False, additional_face_points=0, additional_holistic_config={}):
    if 'static_image_mode' not in additional_holistic_config:
        additional_holistic_config['static_image_mode'] = False
    holistic = mp_holistic.Holistic(**additional_holistic_config)
    try:
        datas = []
        confs = []
        for i, frame in enumerate(tqdm(frames, disable=not progress)):
            results = holistic.process(frame)
            print(f"results[{i}]: {results}")
    finally:
        holistic.close()
    return results




for element in results.left_hand_landmarks.landmark:
    print(f"\nelement ({type(element)}): \n{dir(element)}\n{element}\n")