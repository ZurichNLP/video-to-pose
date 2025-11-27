import openpifpaf
from openpifpaf.network import factory as network_factory
import imageio
import os
import PIL
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.getcwd()
data_dir = f"{base_dir}/data" 
visualization_dir = f"{base_dir}/vis"

class OpenpifpafEstimator():
    def __init__(self):
        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
    
    def estimate(self, video_path, frames):
      poses = []
      for frame in frames:
        pil_im = PIL.Image.fromarray(frame).convert('RGB')
        predictions, gt_anns, image_meta = self.predictor.pil_image(pil_im)
        poses.append(predictions)
      return poses

    def visualize(self, video_name, frames, poses):
      out_path = os.path.join(visualization_dir, f"{video_name}")
      writer = imageio.get_writer(out_path, fps=25, codec='libx264')
      annotation_painter = openpifpaf.show.AnnotationPainter()

      for frame, predictions in zip(frames, poses):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(frame)
        ax.axis('off')

        annotation_painter.annotations(ax, predictions)
        fig.canvas.draw()
        annotated_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        annotated_frame = annotated_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        writer.append_data(annotated_frame)
      writer.close()
      print(f"Saved visualization to: {out_path}")

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
    estimator_name = "openpifpaf"
    print(f"Beginning pose estimation with {estimator_name}")

    estimator = OpenpifpafEstimator()
    for video_name in os.listdir(data_dir):

        # get full path as a string
        video_path = os.path.join(data_dir, video_name)
        frames = load_video_frames(video_path)
        print("\n\nAttempting to estimate pose for video: " + video_name)   

        poses = estimator.estimate(video_path, frames)
        estimator.visualize(video_name, frames, poses)

        print(f"Estimation and visualization for video {video_name} is complete.\n\n")
    print(f"Estimation with {estimator_name} is complete.")

if __name__ == "__main__":
    main()