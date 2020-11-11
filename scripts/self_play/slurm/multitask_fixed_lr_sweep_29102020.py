#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "multitask_fixed_sweep_291020"

time = time.strftime("%Y%m%d-%H%M%S")
job_output_dir = os.path.join(SAVE_DIR, f"{JOB_SWEEP_NAME}-{time}")
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

exp_tuple_list = [
    (80, "rnn", "coach", 5e-7),
    (80, "rnn", "coach", 1e-7),
    (80, "rnn", "coach", 2e-6),
    (80, "rnn", "coach", 3e-6),
    (80, "rnn", "coach", 4e-6),
    (80, "rnn", "coach", 5e-6),
    (80, "rnn", "executor", 5e-7),
    (80, "rnn", "executor", 1e-7),
    (80, "rnn", "executor", 2e-6),
    (80, "rnn", "executor", 3e-6),
    (80, "rnn", "executor", 4e-6),
    (80, "rnn", "executor", 5e-6),
    (80, "rnn", "executor", 7e-6),
    (80, "rnn", "executor", 8e-6),
    (80, "rnn", "executor", 9e-6),
    (80, "zero", "executor", 5e-7),
    (80, "zero", "executor", 1e-7),
    (80, "zero", "executor", 2e-6),
    (80, "zero", "executor", 3e-6),
    (80, "zero", "executor", 4e-6),
    (80, "zero", "executor", 5e-6),
    (80, "zero", "executor", 7e-6),
    (80, "zero", "executor", 8e-6),
    (80, "zero", "executor", 9e-6),
    (80, "rnn", "both", 5e-7),
    (80, "rnn", "both", 1e-7),
    (80, "rnn", "both", 2e-6),
    (80, "rnn", "both", 3e-6),
    (80, "rnn", "both", 4e-6),
    (80, "rnn", "both", 5e-6),
    (80, "rnn", "both", 7e-6),
    (80, "rnn", "both", 8e-6),
    (80, "rnn", "both", 9e-6),
]


for rule, rnn, train_mode, lr in exp_tuple_list:

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
        "--wandb_dir /home/gridsan/apjacob/minirts/wandb/ "
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
