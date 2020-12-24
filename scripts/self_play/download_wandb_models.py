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
from multitask_pop_eval_2 import experiment_list_2, experiment_dict_2
from multitask_pop_eval_3 import *
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
    main_exp_list = experiment_list_3
    main_exp_dict = experiment_dict_3
    print("Downloading files for the following experiments: ", main_exp_list)
    exp_code = args.experiment_code
    eval_exp_list = [main_exp_list[exp_code]] if exp_code != -1 else main_exp_list

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

        for (sub_exp_name, sub_exp_dict) in main_exp_dict[exp_name].items():
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

            if sub_exp_name != "ft_coach[21]-bc" and sub_exp_name.startswith(
                "ft_coach[21]"
            ):
                coaches = [coaches[0]] * len(execs) + [coaches[1]] * len(execs)
                execs = execs * 2
            if sub_exp_name != "ft_coach[14]-bc" and sub_exp_name.startswith(
                "ft_coach[14]"
            ):
                coaches = [coaches[0]] * len(execs)
                execs = execs
                num_total_sub_exps = len(execs)
            if sub_exp_name != "ft_coach[7]-bc" and sub_exp_name.startswith(
                "ft_coach[7]"
            ):
                coaches = [coaches[0]] * len(execs)
                execs = execs

            num_sub_exps = 0
            model_paths = []
            for (coach, executor) in zip(
                coaches, execs
            ):  ## Do we want to check if coaches == execs?
                if coaches == execs and coach != executor:
                    continue

                print(f"Experiment number: {num_sub_exps}")
                coach_model_name, args.coach1 = get_coach_path(
                    coach, coach_variant=coach_variant
                )
                if random_coach:
                    args.coach_random_init = True
                else:
                    args.coach_random_init = False

                random_number = randint(1, 100000)

                exec_model_name, args.executor1 = get_executor_path(
                    executor, exec_variant=exec_variant
                )
                coach_path = args.coach1
                executor_path = args.executor1

                print(f"Coach Path: {args.coach1}")
                print(f"Executor Path: {args.executor1}")

                if "cloned" not in coach:
                    new_coach_fn = f"exp-{num_sub_exps}-{random_number}-{coach_model_name}-{os.path.basename(coach_path)}"
                    new_coach_fp = os.path.join(
                        os.path.dirname(coach_path), new_coach_fn
                    )
                    new_coach_params_fp = os.path.join(
                        os.path.dirname(coach_path), f"{new_coach_fn}.params"
                    )
                    os.rename(coach_path, new_coach_fp)
                    os.rename(f"{coach_path}.params", new_coach_params_fp)

                    copied_coach_path = shutil.copy2(new_coach_fp, model_save_dir)

                    copied_coach_param_path = shutil.copy2(
                        new_coach_params_fp, model_save_dir
                    )
                else:
                    copied_coach_path = coach_path

                if "cloned" not in executor:
                    new_executor_fn = f"exp-{num_sub_exps}-{random_number}-{exec_model_name}-{os.path.basename(executor_path)}"
                    new_executor_fp = os.path.join(
                        os.path.dirname(executor_path), new_executor_fn
                    )
                    new_executor_params_fp = os.path.join(
                        os.path.dirname(executor_path), f"{new_executor_fp}.params"
                    )
                    os.rename(executor_path, new_executor_fp)
                    os.rename(f"{executor_path}.params", new_executor_params_fp)

                    copied_executor_path = shutil.copy2(new_executor_fp, model_save_dir)

                    executor_param_path = shutil.copy2(
                        new_executor_params_fp, model_save_dir
                    )
                else:
                    copied_executor_path = executor_path

                model_paths.append(
                    {"coach": copied_coach_path, "executor": copied_executor_path}
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
