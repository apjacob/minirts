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
from itertools import groupby
from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device, ResultStat, Logger
from best_models import best_executors, best_coaches
from tqdm import tqdm
from game import *
import wandb

model_dicts = {
    "both_finetuned_rule80": {
        "rule": 80,
        "model": "decent-bush-217",
        "run_path": "apjacob/adapt-minirts/250vw2ty",
        "best_exec": "best_exec_checkpoint_421.pt",
        "best_coach": "best_coach_checkpoint_421.pt",
    },
    "both_finetuned_rule12": {
        "rule": 12,
        "model": "sandy-forest-196",
        "run_path": "apjacob/adapt-minirts/1zk0an30",
        "best_exec": "best_exec_checkpoint_1841.pt",
        "best_coach": "best_coach_checkpoint_1841.pt",
    },
    "both_finetuned_rule7": {
        "rule": 7,
        "model": "vibrant-monkey-195",
        "run_path": "apjacob/adapt-minirts/1czxzs7k",
        "best_exec": "best_exec_checkpoint_581.pt",
        "best_coach": "best_coach_checkpoint_581.pt",
    },
    "hier_exec_finetuned_rule80": {
        "rule": 80,
        "model": "smooth-silence-209",
        "run_path": "apjacob/adapt-minirts/3e2vpwt1",
        "best_exec": "best_exec_checkpoint_1501.pt",
        "best_coach": "best_coach_checkpoint_1501.pt",
    },
    "hier_exec_finetuned_rule21": {
        "rule": 21,
        "model": "glowing-violet-207",
        "run_path": "apjacob/adapt-minirts/33hm1jco",
        "best_exec": "best_exec_checkpoint_1381.pt",
        "best_coach": "best_coach_checkpoint_1381.pt",
    },
    "hier_exec_finetuned_rule14": {
        "rule": 14,
        "model": "multitask-fixed_selfplay-exec14-1aqqm60r-",
        "run_path": "apjacob/adapt-minirts/1aqqm60r",
        "best_exec": "best_exec_checkpoint_1821.pt",
        "best_coach": "best_coach_checkpoint_1821.pt",
    },
    "hier_coach_finetuned_rule21": {
        "rule": 21,
        "model": "icy-deluge-205",
        "run_path": "apjacob/adapt-minirts/2w738s03",
        "best_exec": "best_exec_checkpoint_1901.pt",
        "best_coach": "best_coach_checkpoint_1901.pt",
    },
    "hier_coach_finetuned_rule80": {
        "rule": 80,
        "model": "dark-vortex-212",
        "run_path": "apjacob/adapt-minirts/1e4cctnv",
        "best_exec": "best_exec_checkpoint_1141.pt",
        "best_coach": "best_coach_checkpoint_1141.pt",
    },
    "both_finetuned_rule3": {
        "rule": 3,
        "model": "glorious-plasma-197",
        "run_path": "apjacob/adapt-minirts/1iy0lk2d",
        "best_exec": "best_exec_checkpoint_1921.pt",
        "best_coach": "best_coach_checkpoint_1921.pt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multitask eval play interpretability")
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
    parser.add_argument(
        "--coach_load_file", type=str, default="/raid/lingo/apjacob/minirts/save/"
    )
    parser.add_argument(
        "--exec_load_file", type=str, default="/raid/lingo/apjacob/minirts/save/"
    )
    parser.add_argument(
        "--model",
        choices=[
            "both_finetuned_rule80",
            "both_finetuned_rule3",
            "behaviour_cloned",
            "both_finetuned_rule7",
            "both_finetuned_rule14",
            "both_finetuned_rule12",
        ],
    )
    parser.add_argument(
        "--inst_dict_path",
        type=str,
        default="/home/ubuntu/minirts/data/dataset/dict.pt",
    )

    args = parser.parse_args()

    return args


def self_play(args):

    wandb.init(project="adapt-minirts", sync_tensorboard=True, dir=args.wandb_dir)
    # run_id = f"multitask-fixed_selfplay-{args.coach1}-{args.executor1}-{args.train_mode}-rule{args.rule}-{args.tag}"
    wandb.run.name = (
        f"multitask-fixed_analyse-int-{wandb.run.id}-{args.coach1}-{args.executor1}"
        f"-{args.train_mode}-rule{args.rule}-{args.tag}"
    )
    # wandb.run.save()
    wandb.config.update(args)

    print("args:")
    pprint.pprint(vars(args))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    output_list = []

    print("Overriding args.model.... with random model")
    # models = ['both_finetuned_rule80',
    #  'both_finetuned_rule3',
    #  'both_finetuned_rule7',
    #  "both_finetuned_rule14",
    #  "both_finetuned_rule12"]
    # models = ['both_finetuned_rule80', 'both_finetuned_rule3', 'behaviour_cloned', 'hier_exec_finetuned_rule21',
    #          "hier_exec_finetuned_rule80", "both_finetuned_rule12", "both_finetuned_rule7",
    #          "hier_exec_finetuned_rule14", "both_finetuned_rule3", "hier_coach_finetuned_rule80", "hier_coach_finetuned_rule21"]

    models = [
        "hier_exec_finetuned_rule21",
        "hier_exec_finetuned_rule80",
        "hier_exec_finetuned_rule14",
    ]

    # models = ["hier_coach_finetuned_rule80", "hier_coach_finetuned_rule21"]

    args.model = random.choice(models)
    print(f"Using model {args.model}")

    print("Overriding args.rule.... with random rule")
    rules = [80, 7, 14, 12, 3]
    args.rule = random.choice(rules)
    print(f"Using rule {args.rule}")

    if args.model == "behaviour_cloned":
        args.coach1 = best_coaches["rnn500"]
        args.executor1 = best_executors["rnn"]
    else:
        print("Reloading coach model.... ")
        print("Reloading executor model.... ")
        model_dict = model_dicts[args.model]
        args.coach1 = wandb.restore(
            model_dict["best_coach"], run_path=model_dict["run_path"]
        ).name
        wandb.restore(
            model_dict["best_coach"] + ".params", run_path=model_dict["run_path"]
        )
        args.executor1 = wandb.restore(
            model_dict["best_exec"], run_path=model_dict["run_path"]
        ).name
        wandb.restore(
            model_dict["best_exec"] + ".params", run_path=model_dict["run_path"]
        )

    args.coach2 = args.coach1
    args.executor2 = args.executor1
    _coach1 = os.path.basename(args.coach1).replace(".pt", "")
    _executor1 = os.path.basename(args.executor1).replace(".pt", "")

    log_name = "multitask-fixed-analyze-int_c1_type={}_c2_type={}__e1_type={}_e2_type={}__lr={}__num_sp={}__num_rb={}_{}_{}".format(
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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"sp_agent coach params: {count_parameters(sp_agent.model.coach)}")
    print(f"sp_agent exec params: {count_parameters(sp_agent.model.executor)}")
    ## Create Save folder:
    working_rule_dir = os.path.join(sp_agent.save_folder, "rules")
    create_working_dir(args, working_rule_dir)

    cur_iter_idx = 1
    rule_idx = args.rule
    game = MultiTaskGame(sp_agent, bc_agent, cur_iter_idx, args, working_rule_dir)
    for r in [80, 40, 20, 21, 14, 7, 3, 12, 13]:
        game.print_rule_desc(r, split="train")

    writer.close()

    print("#" * 40)
    print("#" * 40)
    print("#" * 40)
    print()
    print("\n\n".join(output_list))
    print()
    print("#" * 40)
    print("#" * 40)
    print("#" * 40)


if __name__ == "__main__":
    global device
    args = parse_args()
    self_play(args)
