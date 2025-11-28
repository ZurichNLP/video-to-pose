import cv2
import os
import imageio

base_dir = os.getcwd()
data_dir = f"{base_dir}/data" 
visualization_dir = f"{base_dir}/vis"
model_dir = f"{base_dir}/models"

class OpenposeEstimator():
    def __init__(self, threshold=0.2, size=(368, 368)):
        self.threshold = threshold
        self.size = size
        os.makedirs(visualization_dir, exist_ok=True)

        self.BODY_PARTS = { 
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }

        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
            ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
            ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
            ["Nose", "LEye"], ["LEye", "LEar"]
        ]
    
    def estimate(self, frames):
        net = cv2.dnn.readNetFromTensorflow(self.model_path)
        results = []

        for frame in frames:
            frameWidth, frameHeight = frame.shape[1], frame.shape[0]
            inp = cv2.dnn.blobFromImage(
                frame, 1.0, self.size, (127.5, 127.5, 127.5),
                swapRB=True, crop=False
            )
            net.setInput(inp)
            out = net.forward()
            out = out[:, :19, :, :]
            points = []

            for i in range(len(self.BODY_PARTS)):
                heatMap = out[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                points.append((int(x), int(y)) if conf > self.threshold else None)
            results.append(points)
        self.poses = results
        return results
    
    def visualize(self, video_name, frames, poses):
        annotated_frames = []
        output_path = f"{visualization_dir}/{video_name}"
        output_path = f"{visualization_dir}/{video_name}"

        for frame, points in zip(frames, poses):
            annotated_frame = self.draw_landmarks(frame, points)
            annotated_frames.append(annotated_frame)
        
        writer = imageio.get_writer(output_path, fps=30)
        for f in annotated_frames:
            writer.append_data(f)  
        writer.close()
        print(f"Visualization saved to: {output_path}")

    def draw_landmarks(self, frame, points):
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] is not None and points[idTo] is not None:
                # Draw line between keypoints
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                # Draw keypoints
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        return frame


def load_video_frames(video_path):
    frames = []
    
    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        raise ValueError(f"Could not open video file: {video_path}\n{e}")

    for frame in reader:
        frames.append(frame)
    
    reader.close()
    return frames

def main():
    estimator_name = "openpose"
    print(f"Beginning pose estimation with estimator {estimator_name}")

    estimator = OpenposeEstimator()
    for video_name in os.listdir(data_dir):

        video_path = os.path.join(data_dir, video_name) # get full path as a string
        frames = load_video_frames(video_path)

        print(f"Estimating for {video_name}:")
        poses = estimator.estimate(frames)
        estimator.visualize(video_name, frames, poses)
    
    print(f"Estimation with {estimator_name} is complete.")

if __name__ == "__main__":
    main()


