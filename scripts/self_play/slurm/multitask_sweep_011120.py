#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "multitask_fixed_sweep_011120"

time = time.strftime("%Y%m%d-%H%M%S")
job_output_dir = os.path.join(SAVE_DIR, f"{JOB_SWEEP_NAME}-{time}")
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

exp_tuple_list = [
    (3, "rnn", "both"),
    (12, "rnn", "both"),
    (80, "rnn", "both"),
    (3, "rnn", "both"),
    (12, "rnn", "both"),
    (80, "rnn", "both"),
    (3, "rnn", "both"),
    (12, "rnn", "both"),
    (80, "rnn", "both"),
    (3, "rnn", "both"),
    (12, "rnn", "both"),
    (80, "rnn", "both"),
    (3, "rnn", "both"),
    (12, "rnn", "both"),
    (80, "rnn", "both"),
    (3, "rnn", "coach"),
    (21, "rnn", "coach"),
    (80, "rnn", "coach"),
    (3, "rnn", "coach"),
    (21, "rnn", "coach"),
    (80, "rnn", "coach"),
    (3, "rnn", "coach"),
    (21, "rnn", "coach"),
    (80, "rnn", "coach"),
    (3, "rnn", "coach"),
    (21, "rnn", "coach"),
    (80, "rnn", "coach"),
    (3, "rnn", "coach"),
    (21, "rnn", "coach"),
    (80, "rnn", "coach"),
    (3, "rnn", "executor"),
    (21, "rnn", "executor"),
    (80, "rnn", "executor"),
    (3, "rnn", "executor"),
    (21, "rnn", "executor"),
    (80, "rnn", "executor"),
    (3, "rnn", "executor"),
    (21, "rnn", "executor"),
    (80, "rnn", "executor"),
    (3, "rnn", "executor"),
    (21, "rnn", "executor"),
    (80, "rnn", "executor"),
    (3, "rnn", "executor"),
    (21, "rnn", "executor"),
    (80, "rnn", "executor"),
    (3, "zero", "executor"),
    (80, "zero", "executor"),
    (3, "zero", "executor"),
    (80, "zero", "executor"),
    (3, "zero", "executor"),
    (80, "zero", "executor"),
    (3, "zero", "executor"),
    (80, "zero", "executor"),
    (3, "zero", "executor"),
    (80, "zero", "executor"),
]


for rule, rnn, train_mode in exp_tuple_list:
    if rnn == "zero" and train_mode != "executor":
        continue

    if train_mode == "both":
        lr = 6e-6
    elif train_mode == "coach":
        lr = 3e-6
    elif train_mode == "executor" and rnn == "zero":
        lr = 7e-6
    elif train_mode == "executor" and rnn == "rnn":
        lr = 5e-6
    else:
        raise EnvironmentError

    job_name = f"rule-{rule}-rnns-{rnn}-train_mode-{train_mode}-lr-{lr}"
    job_file_name = os.path.join(job_output_dir, f"{job_name}.job")
    job_log_file = os.path.join(job_output_dir, f"{job_name}.log")

    python_command = (
        "python -u /home/gridsan/apjacob/minirts/scripts/self_play/multitask-fixed.py --coach1 rnn500 "
        "--coach2 rnn500 "
        f"--executor1 {rnn} "
        f"--executor2 {rnn} "
        f"--lr {lr} "
        "--train_epochs 1800 "
        "--sampling_freq 1.0 "
        "--seed 777 "
        "--tb_log 1 "
        "--save_folder /home/gridsan/apjacob/save "
        "--rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ "
        "--wandb_dir /home/gridsan/apjacob/wandb/ "
        "--pg ppo --ppo_epochs 4 --train_batch_size 32 "
        f"--num_rb=25 --num_sp=0 --train_mode={train_mode} "
        f"--rule {rule} > {job_log_file}"
    )
    print(f"Command: {python_command}")

    job_output = os.path.join(job_output_dir, f"{job_name}")
    file = SLURM_TEMPLATE.format(
        job_output=job_output, job_name=job_name, python_command=python_command
    )

    with open(job_file_name, "w") as f:
        f.write(file.strip())

    subprocess.call(["LLsub", job_file_name])
