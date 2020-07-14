#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --qos=high
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=20
#SBATCH --constraint=xeon-g6
#SBATCH --job-name="multitask-fixed-sweep"
#SBATCH -a 1-8
#SBATCH -o multitask-fixed.out-%j-%a

# Initialize Modules
source /etc/profile

# Load cuda Module
module load cuda/10.0

# Activate conda env
conda activate minirts

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

python top5each.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
