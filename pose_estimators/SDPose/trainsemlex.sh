#!/usr/bin/bash -l

### SBATCH parameters for GPU job
#SBATCH --time=0-160:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=16G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=4          ## 2 CPUs
#SBATCH --gpus=L4:1                ## 1 T4 GPU
#SBATCH --job-name=SDpose_semlex        ## Job name
#SBATCH --output=SDpose_semlex_%j.out   ## Output file (%j = job ID)
#SBATCH --error=SDpose_semlex_%j.err    ## Error file (%j = job ID)

# Load environment modules
# Load environment modules
module load miniforge3
# Activate conda environment
source activate SDPose
# pip install decord
# pip install -r requirements.txt
# Run the Python program
python conbatch_semlex.py