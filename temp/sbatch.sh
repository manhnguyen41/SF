#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/model_ver1.txt      # Output file
#SBATCH --error=log/model_ver1.txt         # Error file
#SBATCH --ntasks=2                         # Number of tasks (processes)
#SBATCH --gpus=1     

sh script/model_v1.sh