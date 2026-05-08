#!/usr/bin/bash -l

#SBATCH --time=0-160:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=16G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=4         ## 2 CPUs
#SBATCH --gpus=L4:1                ## 1 T4 GPU
#SBATCH --job-name=vitpose_popsign        ## Job name
#SBATCH --output=vitpose_popsign_%j.out   ## Output file (%j = job ID)
#SBATCH --error=vitpose_popsign_%j.err    ## Error file (%j = job ID)

module load miniforge3
# Activate conda environment
source activate vitpose

# Run the Python program
python vitpose_batch_processor.py
