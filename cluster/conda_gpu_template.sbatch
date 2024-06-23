#!/bin/sh

#SBATCH --job-name=NAME
#SBATCH --time=HOURS:MINUTES:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=CPUS
#SBATCH --gpus-per-task=GPUS
#SBATCH --mem=MEMORY
#SBATCH --partition=PARTITION
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate thesis

srun COMMAND

conda deactivate
