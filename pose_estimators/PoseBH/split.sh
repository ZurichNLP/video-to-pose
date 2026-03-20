#!/usr/bin/bash -l

#SBATCH --time=0-10:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=30G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=1         ## 2 CPUs
#SBATCH --job-name=posebh_split       ## Job name
#SBATCH --output=posebh_split_%j.out   ## Output file (%j = job ID)
#SBATCH --error=posebh_split_%j.err    ## Error file (%j = job ID)

module load miniforge3
# Activate conda environment
source activate posebh


# Run the Python program
python tools/model_split.py --source weights/posebh/huge.pth
