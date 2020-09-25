#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "multitask_lifelong_learning_sweep_two"

time = time.strftime("%Y%m%d-%H%M%S")
job_output_dir = os.path.join(SAVE_DIR, f"{JOB_SWEEP_NAME}-{time}")
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

# Sweep params
lifelong_series = ["80 3 80", "80 3 12", "80 3 14", "80 12 80", "3 12 13", "3 12 3"]
rnns = ["zero", "rnn"]
train_modes = ["coach", "executor", "both"]
lrs = [1e-5, 5e-5]

for lifelong_serie, rnn, train_mode, lr in itertools.product(lifelong_series, rnns, train_modes, lrs):
    log_series = ",".join(lifelong_serie.split(" "))
    job_name = f"series-{log_series}-rnns-{rnn}-train_mode-{train_mode}-200"
    job_file_name = os.path.join(job_output_dir, f"{job_name}.job")
    job_log_file = os.path.join(job_output_dir, f"{job_name}.log")

    if rnn == "zero" and train_mode != "executor":
        continue

    python_command = "python -u /home/gridsan/apjacob/minirts/scripts/self_play/multitask-lifelong.py --coach1 rnn500 " \
              "--coach2 rnn500 " \
              f"--executor1 {rnn} " \
              f"--executor2 {rnn} " \
              f"--lr {lr} " \
              "--train_epochs 200 " \
              "--sampling_freq 1.0 " \
              "--seed 777 " \
              "--tb_log 1 " \
              "--save_folder /home/gridsan/apjacob/save " \
              "--rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ " \
              "--wandb_dir /home/gridsan/apjacob/wandb/ " \
              "--pg ppo --ppo_epochs 4 --train_batch_size 32 " \
              f"--num_rb=25 --num_sp=0 --train_mode={train_mode} --rule_series {lifelong_serie} > {job_log_file}"

    print(f"Command: {python_command}")

    job_output = os.path.join(job_output_dir, f"{job_name}")
    file = SLURM_TEMPLATE.format(job_output=job_output,
                                 job_name=job_name,
                                 python_command=python_command)

    with open(job_file_name, "w") as f:
        f.write(file.strip())

    subprocess.call(["LLsub", job_file_name])
