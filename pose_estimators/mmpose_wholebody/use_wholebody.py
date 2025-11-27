import os
import cv2
import numpy as np
from pathlib import Path
from mmpose.apis import MMPoseInferencer
from base_estimator import BasePoseEstimator
import mmcv, mmengine, mmdet, mmpose

base_dir = os.getcwd()
data_dir = f"{base_dir}/data" 
visualization_dir = f"{base_dir}/vis"
    
def estimate_and_visualize(self, video_path):

    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer('wholebody')

    result_generator = inferencer(
        video_path,
        show=False, 
        save_vis=True, 
        return_vis=True,
        save_out_video=True,
        out_dir=visualization_dir)  
    results = [result for result in result_generator]

    self.results = results

    vis = results[0]["visualization"]

def main():
    print(f"Beginning pose estimation with mmpose-wholebody.")

    estimator = WholebodyEstimator()
    for video_name in os.listdir(data_dir):
        print("\n\nAttempting to estimate pose for video: " + video_name) 
        video_path = os.path.join(data_dir, video_name) # get full path as a string  
        poses = estimate_and_visualize(video_path)
        print(f"Estimation and visualization for video {video_name} is complete.\n\n")

    print(f"Pose estimation with mmpose-wholebody is complete.")
    
if __name__ == "__main__":
    main()


