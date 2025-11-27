# MMPose Wholebody

## 1. Create and activate environment
```

conda create -y --prefix YOUR_VENV_PATH/mmpose python=3.8
conda activate YOUR_VENV_PATH/mmpose
```

## 2. Install dependencies
```
./install-mmpose.sh
```

## 3. Run
Place your desired videos in the /data folder. 
The resulting visualizations will be created in the /vis folder. 
```
python use_wholebody.py
```

