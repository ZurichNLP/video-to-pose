# Get an interactive session with gpu and cuda enabled: ej. srun --pty --cpus-per-task=8 --constraint=A100 --partition=lowprio --time=05:00:00 --mem-per-gpu=80G --gres=gpu:1 bash -l
module purge
module load a100
module load cuda/11.8.0
module load mamba/24.9.0-0

export CONDA_ENVS_PATH=/home/gsantm/data/conda/envs # Where I want to store the created environment
export MAMBA_ROOT_PREFIX=/home/gsantm/data/conda

mkdir -p /home/gsantm/data/conda/envs

mamba create -n smplestx python=3.8 -y
source activate smplestx


mamba install pytorch==1.12.0 torchvision==0.13.0 \
              torchaudio==0.12.0 cudatoolkit=11.3 \
              -c pytorch -y


cd /home/gsantm/repositories/SMPLest-X
pip install -r requirements.txt

conda install conda-forge::mesalib