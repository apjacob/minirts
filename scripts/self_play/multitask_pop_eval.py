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
from random import randint
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
from pop_utils import model_dicts
from datetime import datetime
from game import *
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Multitask population eval play")
    parser.add_argument("--seed", type=int, default=777)

    parser.add_argument("--num_sp", type=int, default=0)
    parser.add_argument("--num_rb", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument(
        "--train_mode", type=str, choices=["executor", "coach", "both"], default="coach"
    )

    parser.add_argument("--sampling_freq", type=float, default=0.4)
    parser.add_argument("--sp_factor", type=int, default=2)
    parser.add_argument("--eval_factor", type=int, default=20)
    parser.add_argument("--max_table_size", type=int, default=100)

    parser.add_argument("--game_per_thread", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_lua = os.path.join(root, "game/game_MC/lua")
    parser.add_argument("--lua_files", type=str, default=default_lua)

    parser.add_argument("--rule_dir", type=str, default="rules/")

    # ai1 option
    parser.add_argument("--frame_skip", type=int, default=50)
    parser.add_argument("--fow", type=int, default=1)
    parser.add_argument("--use_moving_avg", type=int, default=1)
    parser.add_argument("--moving_avg_decay", type=float, default=0.98)
    parser.add_argument("--num_resource_bins", type=int, default=11)
    parser.add_argument("--resource_bin_size", type=int, default=50)
    parser.add_argument("--max_num_units", type=int, default=50)
    parser.add_argument("--num_prev_cmds", type=int, default=25)
    # TOOD: add max instruction span

    parser.add_argument("--max_raw_chars", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--inst_mode", type=str, default="full"
    )  # can be full/good/better

    # game option
    parser.add_argument("--max_tick", type=int, default=int(2e4))
    parser.add_argument("--no_terrain", action="store_true")
    parser.add_argument("--resource", type=int, default=500)
    parser.add_argument("--resource_dist", type=int, default=4)
    parser.add_argument("--fair", type=int, default=0)
    parser.add_argument("--save_replay_freq", type=int, default=0)
    parser.add_argument("--save_replay_per_games", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="matches2/dev")

    parser.add_argument("--rule", type=int, default=80)
    parser.add_argument("--opponent", type=str, default="rb")
    parser.add_argument(
        "--wandb_dir", type=str, default="/raid/lingo/apjacob/minirts/wandb"
    )
    parser.add_argument(
        "--save_folder", type=str, default="/raid/lingo/apjacob/minirts/save"
    )
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--tb_log", type=int, default=1)
    # full vision
    parser.add_argument("--cheat", type=int, default=0)

    parser.add_argument("--coach1", type=str, default="")
    parser.add_argument("--executor1", type=str, default="")

    parser.add_argument("--coach2", type=str, default="")
    parser.add_argument("--executor2", type=str, default="")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--pg", type=str, default="ppo")
    parser.add_argument("--ppo_eps", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=3)

    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--experiment_code", type=int, default=-1)
    parser.add_argument("--model_json", type=str, default=None)
    parser.add_argument(
        "--inst_dict_path",
        type=str,
        default="/home/ubuntu/minirts/data/dataset/dict.pt",
    )
    parser.add_argument("--coach_random_init", type=bool, default=False)
    parser.add_argument(
        "--eval_folder", type=str, default="/raid/lingo/apjacob/minirts/save"
    )
    args = parser.parse_args()

    return args


model_dicts = model_dicts


experiment_dict = {
    "Drift measured with the cloned coach": {
        "bc-bc": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "bc-ft_both[80]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_both[80]"],
            "env": 80,
        },
        "bc-ft_both[12]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_both[12]"],
            "env": 80,
        },
        "bc-ft_pop[80,40,20]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_pop[80,40,20]"],
            "env": 80,
        },
        "bc-ft_pop[3,12,13]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_pop[3,12,13]"],
            "env": 80,
        },
    },
    "Drift measured with a fine-tuned coach on a new rule": {
        "ft_coach[21]-bc": {
            "coach": model_dicts["ft_coach[21]"],
            "executor": model_dicts["bc"],
            "env": 21,
        },
        "ft_coach[21]-ft_both[80]": {
            "coach": model_dicts["ft_coach[21]"],
            "executor": model_dicts["ft_both[80]"],
            "env": 21,
        },
        "ft_coach[21]-ft_pop[3,12,13]": {
            "coach": model_dicts["ft_coach[21]"],
            "executor": model_dicts["ft_pop[3,12,13]"],
            "env": 21,
        },
        "ft_coach[21]-ft_pop[80,40,20]": {
            "coach": model_dicts["ft_coach[21]"],
            "executor": model_dicts["ft_pop[80,40,20]"],
            "env": 21,
        },
    },
    "Drift measured with the cloned exec": {
        "bc-bc": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "ft_both[80]-bc": {
            "coach": model_dicts["ft_both[80]"],
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "ft_pop[80,40,20]_80-bc": {
            "coach": {
                "coach": model_dicts["ft_pop[80,40,20]"],
                "variant": 80,
                "random": False,
            },
            "executor": model_dicts["bc"],
            "env": 80,
        },
    },
    "Drift measured with a fine-tuned exec on a new rule": {
        "bc-ft_hier_exec[21]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_hier_exec[21]"],
            "env": 21,
        },
        "ft_both[80]-ft_hier_exec[21]": {
            "coach": model_dicts["ft_both[80]"],
            "executor": model_dicts["ft_hier_exec[21]"],
            "env": 21,
        },
        "ft_pop[80,40,20]_80-ft_hier_exec[21]": {
            "coach": {
                "coach": model_dicts["ft_pop[80,40,20]"],
                "variant": 80,
                "random": False,
            },
            "executor": model_dicts["ft_hier_exec[21]"],
            "env": 21,
        },
    },
    "Drift measured with a random coach": {
        "random-bc": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "random-ft_pop[80,40,20]": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["ft_pop[80,40,20]"],
            "env": 80,
        },
        "random-ft_both[80]": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["ft_both[80]"],
            "env": 80,
        },
        "random-ft_hier_exec[80]": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["ft_hier_exec[80]"],
            "env": 80,
        },
    },
    "Performance measured on the original rules": {
        "bc-bc": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "_-bc_zero": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["bc_zero"],
            "env": 80,
        },
        "_-ft_zero[80]": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["ft_zero[80]"],
            "env": 80,
        },
        "bc-ft_hier_exec[80]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_hier_exec[80]"],
            "env": 80,
        },
        "ft_coach[80]-bc": {
            "coach": model_dicts["ft_coach[80]"],
            "executor": model_dicts["bc"],
            "env": 80,
        },
        "ft_both[80]-ft_both[80]": {
            "coach": model_dicts["ft_both[80]"],
            "executor": model_dicts["ft_both[80]"],
            "env": 80,
        },
        "ft_pop[80,40,20]_80-ft_pop[80,40,20]": {
            "coach": {
                "coach": model_dicts["ft_pop[80,40,20]"],
                "variant": 80,
                "random": False,
            },
            "executor": {
                "executor": model_dicts["ft_pop[80,40,20]"],
                "variant": 80,
            },
            "env": 80,
        },
    },
    "Performance when adapting to new rules": {
        "bc-bc": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["bc"],
            "env": 3,
        },
        "_-bc_zero": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["bc_zero"],
            "env": 3,
        },
        "_-ft_zero[3]": {
            "coach": {
                "coach": model_dicts["bc"],
                "variant": None,
                "random": True,
            },
            "executor": model_dicts["ft_zero[3]"],
            "env": 3,
        },
        "bc-ft_hier_exec[3]": {
            "coach": model_dicts["bc"],
            "executor": model_dicts["ft_hier_exec[3]"],
            "env": 3,
        },
        "ft_coach[3]-bc": {
            "coach": model_dicts["ft_coach[3]"],
            "executor": model_dicts["bc"],
            "env": 3,
        },
        "ft_both[3]-ft_both[3]": {
            "coach": model_dicts["ft_both[3]"],
            "executor": model_dicts["ft_both[3]"],
            "env": 3,
        },
        "ft_pop[3,12,13]_3-ft_pop[3,12,13]": {
            "coach": {
                "coach": model_dicts["ft_pop[3,12,13]"],
                "variant": 3,
                "random": False,
            },
            "executor": {
                "executor": model_dicts["ft_pop[3,12,13]"],
                "variant": 3,
            },
            "env": 3,
        },
    },
}

experiment_list = [
    "Drift measured with the cloned coach",
    "Drift measured with a fine-tuned coach on a new rule",
    "Drift measured with the cloned exec",
    "Drift measured with a fine-tuned exec on a new rule",
    "Drift measured with a random coach",
    "Performance measured on the original rules",
    "Performance when adapting to new rules",
]


def get_coach_path(coach, coach_variant=None):
    if "cloned" in coach:
        coach_path = coach["best_coach"]
    else:
        coach_str = (
            "best_coach" if coach_variant is None else f"best_coach_{coach_variant}"
        )
        coach_path = wandb.restore(coach[coach_str], run_path=coach["run_path"]).name
        wandb.restore(coach[coach_str] + ".params", run_path=coach["run_path"])

    return coach_path


def get_executor_path(executor, exec_variant=None):
    if "cloned" in executor:
        exec_path = executor["best_exec"]
    else:
        exec_str = "best_exec" if exec_variant is None else f"best_exec_{exec_variant}"
        exec_path = wandb.restore(
            executor[exec_str], run_path=executor["run_path"]
        ).name
        wandb.restore(
            executor[exec_str] + ".params",
            run_path=executor["run_path"],
        )

    return exec_path


def self_play(args):

    wandb.init(
        project="adapt-minirts-pop-eval", sync_tensorboard=True, dir=args.wandb_dir
    )
    # run_id = f"multitask-fixed_selfplay-{args.coach1}-{args.executor1}-{args.train_mode}-rule{args.rule}-{args.tag}"
    date = datetime.date(datetime.now())
    wandb.run.name = f"multitask-pop-eval-{wandb.run.id}-{date}-{args.tag}"
    # wandb.run.save()
    wandb.config.update(args)

    # print("args:")
    # pprint.pprint(vars(args))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    result_dict = {}

    exp_code = args.experiment_code
    eval_exp_list = [experiment_list[exp_code]] if exp_code != -1 else experiment_list
    partial_json_save_dir = os.path.join(args.eval_folder, f"eval-{date}-{args.tag}")

    if os.path.exists(partial_json_save_dir):
        print("Attempting to create an existing folder.. hence skipping...")
    else:
        os.makedirs(partial_json_save_dir)

    if args.model_json is not None:
        if os.path.exists(args.model_json):
            print("Using model json dictionary...")
            with open(os.path.join(args.model_json, "model_paths.json")) as f:
                model_json = json.load(f)
        else:
            FileNotFoundError("Model json dict cannot be found...")
    else:
        model_json = None

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

            rule = sub_exp_dict["env"]
            num_sub_exps = 0
            win_rates = []
            num_total_sub_exps = min(len(coaches), len(execs))
            for (coach, executor) in zip(
                coaches, execs
            ):  ## Do we want to check if coaches == execs?
                if coaches == execs and coach != executor:
                    continue

                print(f"Experiment number: {num_sub_exps}")
                if model_json is not None and args.model_json is not None:
                    assert num_total_sub_exps == len(
                        model_json[exp_name][sub_exp_name]["model_paths"]
                    ), "Number of sub-exp mismatch."

                    args.coach1 = model_json[exp_name][sub_exp_name]["model_paths"][
                        num_sub_exps
                    ]["coach"]
                else:
                    args.coach1 = get_coach_path(coach, coach_variant=coach_variant)

                if random_coach:
                    args.coach_random_init = True
                else:
                    args.coach_random_init = False

                if model_json is not None and args.model_json is not None:
                    args.executor1 = model_json[exp_name][sub_exp_name]["model_paths"][
                        num_sub_exps
                    ]["executor"]
                else:
                    args.executor1 = get_executor_path(
                        executor, exec_variant=exec_variant
                    )

                args.rule = rule

                args.coach2 = args.coach1
                args.executor2 = args.executor1
                _coach1 = os.path.basename(args.coach1).replace(".pt", "")
                _executor1 = os.path.basename(args.executor1).replace(".pt", "")

                log_name = "multitask-pop-analyze_c1_type={}_c2_type={}__e1_type={}_e2_type={}__lr={}__num_sp={}__num_rb={}_{}_{}".format(
                    _coach1,
                    args.coach2,
                    _executor1,
                    args.executor2,
                    args.lr,
                    args.num_sp,
                    args.num_rb,
                    args.tag,
                    random.randint(1111, 9999),
                )
                writer = SummaryWriter(comment=log_name)

                logger_path = os.path.join(args.save_dir, "train.log")

                sys.stdout = Logger(logger_path)

                device = torch.device("cuda:%d" % args.gpu)

                sp_agent = Agent(
                    coach=args.coach1,
                    executor=args.executor1,
                    device=device,
                    args=args,
                    writer=writer,
                    trainable=True,
                    exec_sample=True,
                    pg=args.pg,
                )

                sp_agent.init_save_folder(wandb.run.name)

                bc_agent = Agent(
                    coach=args.coach2,
                    executor=args.executor2,
                    device=device,
                    args=args,
                    writer=writer,
                    trainable=False,
                    exec_sample=False,
                )

                print("Progress: ")
                ## Create Save folder:
                working_rule_dir = os.path.join(sp_agent.save_folder, "rules")
                create_working_dir(args, working_rule_dir)

                cur_iter_idx = 1
                rules = [args.rule]

                print("Current rule: {}".format(rules[0]))
                game = MultiTaskGame(
                    sp_agent, bc_agent, cur_iter_idx, args, working_rule_dir
                )
                result = game.analyze_rule_games(
                    cur_iter_idx,
                    rules,
                    "train",
                    viz=args.viz,
                    num_games=args.num_games,
                )
                game.terminate()
                del game

                writer.close()
                win_rate = result[args.rule]["win"]

                win_rates.append(win_rate * 100)
                num_sub_exps += 1

            sub_exp_result_dict[sub_exp_name] = {
                "win_rate": win_rates,
                "Win_rate mean": np.mean(win_rates),
                "Win_rate variance ": np.var(win_rates),
                "Num total trials": num_sub_exps,
            }

            print("++" * 50)
            pprint.pprint(sub_exp_result_dict[sub_exp_name])
            print("++" * 50)

        result_dict[exp_name] = sub_exp_result_dict
        print("--" * 50)
        print("Results so far: ")
        print("--" * 50)
        pprint.pprint(result_dict)
        print("--" * 50)

    print("Final Results: ")
    print("##" * 50)
    pprint.pprint(result_dict)
    print("##" * 50)

    print("Saving result jsons...")
    code = exp_code if exp_code != -1 else "all"
    random_number = randint(1, 100000)
    with open(
        os.path.join(partial_json_save_dir, f"partial-{random_number}-{code}.json"), "w"
    ) as fp:
        json.dump(result_dict, fp)


if __name__ == "__main__":
    global device
    args = parse_args()
    self_play(args)
