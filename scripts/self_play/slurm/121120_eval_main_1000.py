#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, SLURM_TEMPLATE_40_CPUS, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "121120_eval_main_1000"

time = time.strftime("%Y%m%d-%H%M%S")
unique_name = f"{JOB_SWEEP_NAME}-{time}"
job_output_dir = os.path.join(SAVE_DIR, unique_name)
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

# Sweep params
experiments = [0, 1, 2, 3, 4, 5, 6]

for exp_code in experiments:

    job_name = f"{exp_code}-exp_code-adapt-minirts"
    job_file_name = os.path.join(job_output_dir, f"{job_name}.job")
    job_log_file = os.path.join(job_output_dir, f"{job_name}.log")

    python_command = (
        "python -u /home/gridsan/apjacob/minirts/scripts/self_play/multitask_pop_eval.py "
        "--seed=777 --num_games=1000 --num_trials=1 "
        "--save_folder /home/gridsan/apjacob/save "
        "--rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ "
        "--wandb_dir /home/gridsan/apjacob/minirts/wandb/ "
        "--model_json=/home/gridsan/apjacob/save/wandb_models_2020-11-11 "
        "--eval_folder=/home/gridsan/apjacob/eval "
        f"--tag={unique_name} "
        f"--inst_dict_path=/home/gridsan/apjacob/minirts/data/dataset/dict.pt --experiment_code={exp_code}"
    )

    print(f"Command: {python_command}")

    job_output = os.path.join(job_output_dir, f"{job_name}")
    file = SLURM_TEMPLATE_40_CPUS.format(
        job_output=job_output, job_name=job_name, python_command=python_command
    )

    with open(job_file_name, "w") as f:
        f.write(file.strip())

    subprocess.call(["LLsub", job_file_name])
