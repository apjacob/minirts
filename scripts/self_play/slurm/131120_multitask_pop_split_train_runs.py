#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "multitask_pop_sweep_111120"

time = time.strftime("%Y%m%d-%H%M%S")
job_output_dir = os.path.join(SAVE_DIR, f"{JOB_SWEEP_NAME}-{time}")
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

# Sweep params
NUM_TRIALS = 1
lifelong_series = ["80 40 20", "3 12 13"]
rnns = ["rnn"]
train_modes = ["both"]
lrs = [1e-6, 3e-6, 5e-6, 6e-6]

for trial in range(NUM_TRIALS):
    for lifelong_serie, rnn, train_mode, lr in itertools.product(
        lifelong_series, rnns, train_modes, lrs
    ):
        log_series = ",".join(lifelong_serie.split(" "))
        job_name = f"series-{log_series}-rnns-{rnn}-train_mode-{train_mode}-lr-{lr}-trial-{trial}"
        job_file_name = os.path.join(job_output_dir, f"{job_name}.job")
        job_log_file = os.path.join(job_output_dir, f"{job_name}.log")

        python_command = (
            "python -u /home/gridsan/apjacob/minirts/scripts/self_play/multitask-pop.py --coach1 rnn500 "
            "--coach2 rnn500 "
            f"--executor1 {rnn} "
            f"--executor2 {rnn} "
            f"--lr {lr} "
            "--train_epochs 600 "
            "--sampling_freq 1.0 "
            "--seed 777 "
            "--tb_log 1 "
            "--split_train True "
            "--save_folder /home/gridsan/apjacob/save "
            "--rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ "
            "--wandb_dir /home/gridsan/apjacob/minirts/wandb/ "
            "--pg ppo --ppo_epochs 4 --train_batch_size 32 "
            f"--num_rb=25 --num_sp=0 --train_mode={train_mode} --rule_series {lifelong_serie} > {job_log_file}"
        )

        print(f"Command: {python_command}")

        job_output = os.path.join(job_output_dir, f"{job_name}")
        file = SLURM_TEMPLATE.format(
            job_output=job_output, job_name=job_name, python_command=python_command
        )

        with open(job_file_name, "w") as f:
            f.write(file.strip())

        subprocess.call(["LLsub", job_file_name])
