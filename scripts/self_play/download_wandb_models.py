# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
import pprint

from set_path import append_sys_path

append_sys_path()

import torch
import random
import json
import tube
from agent import Agent
from pytube import DataChannelManager
from common_utils import StateActionBuffer
from torch.utils.tensorboard import SummaryWriter
import minirts
import numpy as np
import pickle
from collections import defaultdict
import torch.optim as optim
from rnn_coach import ConvRnnCoach
from onehot_coach import ConvOneHotCoach
from rnn_generator import RnnGenerator
from itertools import groupby, product
from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device, ResultStat, Logger
from best_models import best_executors, best_coaches
from tqdm import tqdm
from multitask_pop_eval import *
from datetime import datetime
from game import *
import wandb
from random import randint
import shutil


def self_play(args):
    # print("args:")
    # pprint.pprint(vars(args))
    wandb.init(
        project="adapt-minirts-pop-eval", sync_tensorboard=True, dir=args.wandb_dir
    )
    # run_id = f"multitask-fixed_selfplay-{args.coach1}-{args.executor1}-{args.train_mode}-rule{args.rule}-{args.tag}"
    wandb.run.name = f"model-download-{wandb.run.id}-{datetime.date(datetime.now())}"
    # wandb.run.save()
    wandb.config.update(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    result_dict = {}

    exp_code = args.experiment_code
    eval_exp_list = [experiment_list[exp_code]] if exp_code != -1 else experiment_list

    model_save_dir = os.path.join(
        args.save_folder,
        f"wandb_models_{datetime.date(datetime.now())}",
    )

    if os.path.exists(model_save_dir):
        print("Attempting to create an existing folder.. hence skipping...")
        raise FileExistsError
    else:
        print(f"Creating save folder: {model_save_dir}")
        os.makedirs(model_save_dir)

    print("Downloading models...")
    for exp_name in eval_exp_list:
        print("#" * 40)
        print("#" * 40)
        print(f"Experiment: {exp_name}")
        print("-" * 40)
        print("-" * 40)
        sub_exp_result_dict = {}

        for (sub_exp_name, sub_exp_dict) in experiment_dict[exp_name].items():
            print("*" * 40)
            print(f"Sub experiment name: {sub_exp_name}")
            print("*" * 40)
            coaches = sub_exp_dict["coach"]

            if "random" in coaches:
                coach_variant = coaches["variant"]
                random_coach = coaches["random"]
                coaches = coaches["coach"]
            else:
                coach_variant = None
                random_coach = False

            execs = sub_exp_dict["executor"]
            if "variant" in execs:
                exec_variant = execs["variant"]
                execs = execs["executor"]
            else:
                exec_variant = None

            num_sub_exps = 0
            model_paths = []
            for (coach, executor) in zip(
                coaches, execs
            ):  ## Do we want to check if coaches == execs?
                if coaches == execs and coach != executor:
                    continue

                print(f"Experiment number: {num_sub_exps}")
                args.coach1 = get_coach_path(coach, coach_variant=coach_variant)
                if random_coach:
                    args.coach_random_init = True
                else:
                    args.coach_random_init = False

                random_number = randint(1, 100000)

                args.executor1 = get_executor_path(executor, exec_variant=exec_variant)
                coach_path = args.coach1
                executor_path = args.executor1

                new_coach_path = shutil.copyfile(
                    coach_path,
                    os.path.join(
                        model_save_dir,
                        f"exp-{num_sub_exps}-{random_number}-{os.path.basename(coach_path)}",
                    ),
                )

                coach_param_path = shutil.copyfile(
                    f"{coach_path}.params",
                    os.path.join(
                        model_save_dir,
                        f"exp-{num_sub_exps}-{random_number}-{os.path.basename(coach_path)}.params",
                    ),
                )

                new_executor_path = shutil.copyfile(
                    executor_path,
                    os.path.join(
                        model_save_dir,
                        f"exp-{num_sub_exps}-{random_number}-{os.path.basename(executor_path)}",
                    ),
                )

                executor_param_path = shutil.copyfile(
                    f"{executor_path}.params",
                    os.path.join(
                        model_save_dir,
                        f"exp-{num_sub_exps}-{random_number}-{os.path.basename(executor_path)}.params",
                    ),
                )

                model_paths.append(
                    {"coach": new_coach_path, "executor": new_executor_path}
                )
                num_sub_exps += 1

                if sub_exp_name == "bc-bc" or sub_exp_name == "random-bc":
                    break

            sub_exp_result_dict[sub_exp_name] = {
                "model_paths": model_paths,
            }

            print("++" * 50)
            pprint.pprint(sub_exp_result_dict[sub_exp_name])
            print("++" * 50)

        result_dict[exp_name] = sub_exp_result_dict

    print(f"Models saved to: {model_save_dir}")
    print(f"Saving model paths to: {model_save_dir}")

    print("Saving model path jsons...")
    with open(os.path.join(model_save_dir, "model_paths.json"), "w") as fp:
        json.dump(result_dict, fp)


if __name__ == "__main__":
    global device
    args = parse_args()
    self_play(args)
