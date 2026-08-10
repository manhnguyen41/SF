#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/loss_func.txt      # Output file
#SBATCH --error=log/loss_func.txt         # Error file
#SBATCH --ntasks=2                         # Number of tasks (processes)
#SBATCH --gpus=1     

sh script/weightedmse.sh