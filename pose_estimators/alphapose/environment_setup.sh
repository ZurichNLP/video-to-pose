

# LOAD MODULES
module load a100
module load cuda/11.8.0
module load anaconda3/2024.02-1
module load mamba/24.9.0-0

# CLONE
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

# ENV
mamba create -n alphapose python=3.10 -y
conda activate alphapose

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PIP_NO_BUILD_ISOLATION=1

# TORCH
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# -------- Pip / Setuptools pin ----------
# Match the versions that behaved well with setup.py develop
say "Pinning pip & setuptools to known-good versions"
python -m pip install --upgrade "pip<25" "setuptools<70" wheel

# -------- PyPI packages (exact pins to match your working env) ----------
say "Installing Python packages (pinned)"
# NOTE: numpy<2 prevents np.float issues inside legacy code (AlphaPose + cython_bbox).
# We pin opencv-python to 4.11.0.86 because >=4.12 pulls numpy>=2.
python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "matplotlib==3.10.7" \
  "scipy==1.15.3" \
  "tqdm==4.67.1" \
  "tensorboardx==2.6.4" \
  "visdom==0.2.4" \
  "terminaltables==3.1.10" \
  "easydict==1.13" \
  "six==1.17.0" \
  "natsort==8.4.0" \
  "timm==0.1.20" \
  "pyyaml==6.0.3" \
  "protobuf==6.33.0" \
  "packaging==25.0" \
  "cython==3.2.0"

python -m pip install "pycocotools==2.0.10" "halpecocotools==0.0.1"

# cython_bbox: the version that compiled & ran
say "Installing cython_bbox (0.1.5)"
# Avoid build isolation so it uses the env’s Cython/Numpy
export PIP_NO_BUILD_ISOLATION=1
python -m pip install --no-build-isolation "cython-bbox==0.1.5"

# -------- Build AlphaPose (like you did) ----------
say "Building AlphaPose with setup.py (develop)"
cd "$REPO_DIR"
# Keep the PIP_NO_BUILD_ISOLATION set so setup_requires resolves via env
export PIP_NO_BUILD_ISOLATION=1
python setup.py build develop # if fails use pip install --no-deps --no-build-isolation .

# -------- Done ----------
say "Environment ready!  Activate with:  conda activate $ENV_NAME"
say "Repo: $REPO_DIR"