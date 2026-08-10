#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/spatial_1.txt      # Output file
#SBATCH --error=log/spatial_1.txt         # Error file
#SBATCH --ntasks=2                         # Number of tasks (processes)
#SBATCH --gpus=1     

sh script/spatial_1.sh