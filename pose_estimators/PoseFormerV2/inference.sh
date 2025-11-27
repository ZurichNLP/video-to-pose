# Get an interactive session with gpu and cuda enabled: ej. srun --pty --cpus-per-task=8 --constraint=A100 --partition=lowprio --time=05:00:00 --mem-per-gpu=80G --gres=gpu:1 bash -l
module purge
module load a100
module load cuda/11.8.0
module load mamba/24.9.0-0


export CONDA_ENVS_PATH=/home/gsantm/data/conda/envs # Where I want to store the created environment
export MAMBA_ROOT_PREFIX=/home/gsantm/data/conda

source activate poseformerv2

export REPO_PATH="/home/gsantm/repositories/PoseFormerV2/"

cd $REPO_PATH

python demo/vis.py --video test.mp4