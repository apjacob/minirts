import os

SLURM_TEMPLATE = """
#!/bin/bash

#SBATCH --partition=normal
#SBATCH --gres=gpu:volta:1
#SBATCH --qos=high
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=20
#SBATCH --constraint=xeon-g6
#SBATCH --job-name={job_name}
#SBATCH -o {job_output}-%j.out
#SBATCH -e {job_output}-%j.err

source ~/.bashrc

# Initialize Modules
source /etc/profile

# Load cuda Module
module load cuda/10.1

# Activate conda env
conda activate minirts2

{python_command}
"""


def create_dir(dir):
    if not os.path.exists(dir):
        print(f"Creating job dir: {dir}")
        os.makedirs(dir)
