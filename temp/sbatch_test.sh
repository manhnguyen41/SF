#!/bin/bash
#SBATCH --job-name=longnd # Job name
#SBATCH --output=error/test.txt      # Output file
#SBATCH --error=log/test.txt         # Error file
#SBATCH --ntasks=2                         # Number of tasks (processes)
#SBATCH --gpus=1

sh run_test.sh