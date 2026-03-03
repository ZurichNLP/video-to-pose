pip install "protobuf>=4.21,<5"
pip install --upgrade mediapipe imageio "imageio[ffmpeg]" opencv-python tensorflow
pip install pose-format
pip install opencv-python

cd models

if [ ! -f hand_landmarker.task ]; then
    echo "Downloading Mediapipe Hands model..."
    wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
else
    echo "Mediapipe Hands model already exists. Skipping download."
fi