#!/usr/bin/env python
import time
import os
import subprocess
import itertools
from utils import SLURM_TEMPLATE, create_dir

SAVE_DIR = "/home/gridsan/apjacob/sweeps"
JOB_SWEEP_NAME = "multitask_zero_sweep_sum"

time = time.strftime("%Y%m%d-%H%M%S")
job_output_dir = os.path.join(SAVE_DIR, f"{JOB_SWEEP_NAME}-{time}")
print(f"Job output directory: {job_output_dir}")

# Make top level directories
create_dir(job_output_dir)

# Sweep params
coach_rule_emb_sizes = [50, 100]
lrs = [1e-5, 5e-6, 1e-6]
rnns = ["rnn"]
train_modes = ["coach", "both"]
update_iters = [9999, 2]

for coach_rule_emb_size, update_iter, lr, rnn, train_mode in \
        itertools.product(coach_rule_emb_sizes, update_iters, lrs, rnns, train_modes):
    job_name = f"coach_emb-{coach_rule_emb_size}-update_iter-{update_iter}-lr-{lr}-rnns-{rnn}-train_mode-{train_mode}"
    job_file_name = os.path.join(job_output_dir, f"{job_name}.job")
    job_log_file = os.path.join(job_output_dir, f"{job_name}.log")

    if rnn == "zero" and (train_mode != "executor" or coach_rule_emb_size > 0) :
        continue

    if coach_rule_emb_size > 0 and train_mode == "executor":
        continue

    # if executor_rule_emb_size > 0 and train_mode == "coach":
    #     continue

    python_command = "python -u /home/gridsan/apjacob/minirts/scripts/self_play/multitask-zero.py --coach1 rnn500 " \
              "--coach2 rnn500 " \
              f"--executor1 {rnn} " \
              f"--executor2 {rnn} " \
              f"--lr {lr} " \
              "--train_epochs 2000 " \
              "--sampling_freq 1.0 " \
              "--seed 777 " \
              "--tb_log 1 " \
              "--save_folder /home/gridsan/apjacob/save " \
              "--rule_dir /home/gridsan/apjacob/minirts/scripts/self_play/rules/ " \
              "--wandb_dir /home/gridsan/apjacob/wandb/ " \
              "--pg ppo --ppo_epochs 4 --train_batch_size 32 " \
              f"--num_rb=25 --num_sp=0 --train_mode={train_mode} " \
              f"--executor_rule_emb_size=0 " \
              f"--coach_rule_emb_size={coach_rule_emb_size} --update_iter={update_iter} > {job_log_file}"

    print(f"Command: {python_command}")

    job_output = os.path.join(job_output_dir, f"{job_name}")
    file = SLURM_TEMPLATE.format(job_output=job_output,
                                 job_name=job_name,
                                 python_command=python_command)

    with open(job_file_name, "w") as f:
        f.write(file.strip())

    subprocess.call(["LLsub", job_file_name])
