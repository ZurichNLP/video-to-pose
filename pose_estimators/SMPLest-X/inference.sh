#!/bin/bash
# Get an interactive session with gpu and cuda enabled: ej. srun --pty --cpus-per-task=8 --constraint=A100 --partition=lowprio --time=05:00:00 --mem-per-gpu=80G --gres=gpu:1 bash -l
module purge
module load a100
module load cuda/11.8.0
module load mamba/24.9.0-0

export CONDA_ENVS_PATH=/home/gsantm/data/conda/envs # Where I want to store the created environment
export MAMBA_ROOT_PREFIX=/home/gsantm/data/conda
export PYOPENGL_PLATFORM=osmesa
export MODEL_DIR="smplest_x_h"
export FILE_NAME="test.mp4"
export FPS=50

source activate smplestx

cd /home/gsantm/repositories/SMPLest-X

pwd

sh scripts/inference.sh ${MODEL_DIR} ${FILE_NAME} ${FPS}