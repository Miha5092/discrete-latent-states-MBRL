#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# students: use #SBATCH --account=Education-EEMCS-Courses-CSE3000
#SBATCH --account=research-eemcs-insy
#SBATCH --partition=PARTITION

# The default run4
#SBATCH --time=HOURS:MINUTES:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default-per-cpu=1024
#SBATCH --cpus-per-task=CPUS

#SBATCH --gpus-per-task=GPUS

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=MEMORY

#PLACEHOLDER1
#PLACEHOLDER2
#PLACEHOLDER3
#PLACEHOLDER4
#PLACEHOLDER5

time srun apptainer exec --nv --bind $(pwd -P) container/container.sif COMMAND
