#!/usr/bin/bash -l

#SBATCH --time=0-160:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=16G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=4         ## 2 CPUs
#SBATCH --gpus=L4:1                ## 1 T4 GPU
#SBATCH --job-name=posebh_citizen        ## Job name
#SBATCH --output=posebh_citizen_%j.out   ## Output file (%j = job ID)
#SBATCH --error=posebh_citizen_%j.err    ## Error file (%j = job ID)

module load miniforge3
# Activate conda environment
source activate posebh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


# Run the Python program
python video_inference_wholebody_mp.py \
    --input_dir /shares/iict-sp2.ebling.cl.uzh/common/ASL_Citizen/videos \
    --output_dir /shares/iict-sp2.ebling.cl.uzh/common/ASL_Citizen/posebh \
    --pose_config configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py \
    --pose_checkpoint weights/posebh/wholebody.pth \
    --use_yolo \
    --yolo_checkpoint yolov8n.pt \
    --device cuda:0 \
    --num_workers 3 \
    --batch_size 4 \
    --queue_size 8 \
    --frame_interval 1 \
    --resume
