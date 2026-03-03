pip uninstall -y numpy mmcv mmpose mmengine torch mmdet

pip install numpy==1.26.4

pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118    # or CPU
pip install mmengine==0.10.3
pip install mmcv==2.1.0
pip install mmpose==1.3.0
pip install mmdet==3.2.0