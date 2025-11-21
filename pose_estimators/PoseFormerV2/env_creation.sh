# Get an interactive session with gpu and cuda enabled: ej. srun --pty --cpus-per-task=8 --constraint=A100 --partition=lowprio --time=05:00:00 --mem-per-gpu=80G --gres=gpu:1 bash -l
module purge
module load a100
module load cuda/11.8.0
module load mamba/24.9.0-0

export CONDA_ENVS_PATH=/home/gsantm/data/conda/envs # Where I want to store the created environment
export MAMBA_ROOT_PREFIX=/home/gsantm/data/conda


mkdir -p /home/gsantm/data/conda/envs

mamba create -y -n poseformerv2 python=3.9
source activate poseformerv2

pip install torch==1.13.0+cu117 \
            torchvision==0.14.0+cu117 \
            torchaudio==0.13.0 \
            --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt