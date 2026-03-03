# Mediapipe Hands

## 1. Create and activate environment
```

conda create -y --prefix YOUR_VENV_PATH/mediapipe-hands python=3.11
conda activate YOUR_VENV_PATH/mediapipe-hands
```

## 2. Install dependencies
After running the below code, you should see a file called hand_landmarker in the /models folder. 
```
cd mediapipehands
./install-mediapipehands.sh
```

## 3. Run
Place your desired videos in the /data folder. 
The resulting visualizations will be created in the /vis folder. 
```
python use_mediapipehands.py
```
