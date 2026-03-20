#!/usr/bin/bash -l

### SBATCH parameters for GPU job
#SBATCH --time=0-10:00:00          ## 7 hours (days-hours:minutes:seconds)
#SBATCH --mem=16G                  ## 16GB RAM
#SBATCH --ntasks=1                 ## Single task
#SBATCH --cpus-per-task=1          ## 2 CPUs
#SBATCH --job-name=SDpose        ## Job name
#SBATCH --output=SDpose_%j.out   ## Output file (%j = job ID)
#SBATCH --error=SDpose_%j.err    ## Error file (%j = job ID)

# Load environment modules
# Load environment modules
module load miniforge3
# Activate conda environment
source activate SDPose
# pip3 install gdown

cd /shares/iict-sp2.ebling.cl.uzh/common/Sem-Lex/

files=(
    "https://drive.google.com/file/d/1jiUasWSGv5lkrBUIRmtCXyMzliClCqXo/view"
    "https://drive.google.com/file/d/1VvrbYgNZe_4fWS5ZdSsHyxOuWHmhisGq/view"
    "https://drive.google.com/file/d/1nVjvgJhjo3lILr5S23p_PsMR7yFdQTrS/view"
)

echo "Starting clean downloads..."

for url in "${files[@]}"; do
    file_id=$(echo "$url" | sed -n 's#.*/d/\([^/]*\)/.*#\1#p')
    clean_url="https://drive.google.com/uc?id=${file_id}"

    echo "Downloading: $clean_url"
    gdown "$clean_url"
done

echo "Downloads completed."