from ultralytics import YOLO
import shutil
import os

base_dir = os.getcwd()
data_dir = f"{base_dir}/data" 
visualization_dir = f"{base_dir}/vis"

class YoloposeEstimator():
    def __init__(self):
        self.model = YOLO("yolo11n-pose.pt")
        os.makedirs(visualization_dir, exist_ok=True)
    
    def estimate_and_visualize(self, video_path):
        poses = self.model.predict(video_path, save=True)

        # Move files to correct directory
        predict_dir = os.path.join(base_dir, "runs", "pose", "predict")
        if os.path.exists(predict_dir):
            for filename in os.listdir(predict_dir):
                src_path = os.path.join(predict_dir, filename)
                dst_path = os.path.join(visualization_dir, filename)
                print(f"Moving {src_path} to {dst_path}")
                shutil.move(src_path, dst_path)

        # Delete incorrect directory
        runs_dir = os.path.join(base_dir, "runs")
        if os.path.exists(runs_dir):
            print(f"Deleting directory: {runs_dir}")
            shutil.rmtree(runs_dir)
        
        return poses

def main():
    estimator = YoloposeEstimator()
    estimator_name = "Yolopose"
    print(f"Beginning pose estimation with {estimator_name}")
    for video_name in os.listdir(data_dir):

        # get full path as a string
        video_path = os.path.join(data_dir, video_name)
        print("\n\nAttempting to estimate pose for video: " + video_name)   

        poses = estimator.estimate_and_visualize(video_path)

        print(f"Estimation and visualization for video {video_name} is complete.\n\n")
    print(f"Estimation with {estimator_name} is complete.")

if __name__ == "__main__":
    main()
