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


def parse_args():
    parser = argparse.ArgumentParser(description="Minirts multitask population-based")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--num_sp", type=int, default=10)
    parser.add_argument("--num_rb", type=int, default=10)
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

    parser.add_argument("--opponent", type=str, default="rb")
    parser.add_argument(
        "--wandb_dir", type=str, default="/raid/lingo/apjacob/minirts/wandb"
    )
    parser.add_argument(
        "--save_folder", type=str, default="/raid/lingo/apjacob/minirts/save"
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--coach_load_file", type=str, default="/raid/lingo/apjacob/minirts/save/"
    )
    parser.add_argument("--coach_reload", type=int, default=0)
    parser.add_argument(
        "--exec_load_file", type=str, default="/raid/lingo/apjacob/minirts/save/"
    )
    parser.add_argument("--exec_reload", type=int, default=0)

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
    parser.add_argument(
        "--rule_series", nargs="+", help="series of rules", required=True
    )
    parser.add_argument("--coach_random_init", type=bool, default=False)
    parser.add_argument("--split_train", type=bool, default=False)
    args = parser.parse_args()

    return args


def self_play(args):

    wandb.init(project="adapt-minirts-pop", sync_tensorboard=True, dir=args.wandb_dir)
    # run_id = f"multitask-fixed_selfplay-{args.coach1}-{args.executor1}-{args.train_mode}-rule{args.rule}-{args.tag}"
    log_series = ",".join(args.rule_series)
    wandb.run.name = (
        f"multitask-pop_selfplay-{wandb.run.id}-{args.coach1}-{args.executor1}"
        f"-{args.train_mode}-rule_series={log_series}-random_coach={args.coach_random_init}-{args.tag}"
    )
    # wandb.run.save()
    wandb.config.update(args)

    print("args:")
    pprint.pprint(vars(args))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Train Mode: {}".format(args.train_mode))

    if args.coach_reload:
        print("Reloading coach model.... ")
        args.coach1 = args.coach_load_file
        _coach1 = os.path.basename(args.coach1).replace(".pt", "")

    else:
        _coach1 = args.coach1
        args.coach1 = best_coaches[args.coach1]

    if args.exec_reload:
        print("Reloading executor model.... ")
        args.executor1 = args.exec_load_file
        _executor1 = os.path.basename(args.executor1).replace(".pt", "")
    else:
        _executor1 = args.executor1
        args.executor1 = best_executors[args.executor1]

    log_name = "multitask-pop_c1_type={}_c2_type={}__e1_type={}_e2_type={}__lr={}__num_sp={}__num_rb={}_coach_random_init={}_{}_{}".format(
        _coach1,
        args.coach2,
        _executor1,
        args.executor2,
        args.lr,
        args.num_sp,
        args.num_rb,
        args.coach_random_init,
        args.tag,
        random.randint(1111, 9999),
    )
    writer = SummaryWriter(comment=log_name)

    args.coach2 = best_coaches[args.coach2]
    args.executor2 = best_executors[args.executor2]

    logger_path = os.path.join(args.save_dir, "train.log")

    sys.stdout = Logger(logger_path)

    device = torch.device("cuda:%d" % args.gpu)

    agent_dict = {}
    for rule in args.rule_series:
        sp_agent = Agent(
            coach=args.coach1,
            executor=args.executor1,
            device=device,
            args=args,
            writer=writer,
            trainable=True,
            exec_sample=True,
            pg=args.pg,
            tag=f"_{rule}",
        )

        ## Sharing executors
        args.executor1 = sp_agent.model.executor

        sp_agent.init_save_folder(wandb.run.name)

        bc_agent = Agent(
            coach=args.coach2,
            executor=args.executor2,
            device=device,
            args=args,
            writer=writer,
            trainable=False,
            exec_sample=False,
            tag=f"_{rule}",
        )

        agent_dict[int(rule)] = {"sp_agent": sp_agent, "bc_agent": bc_agent}

    print("Progress: ")
    ## Create Save folder:
    working_rule_dir = os.path.join(sp_agent.save_folder, "rules")
    create_working_dir(args, working_rule_dir)

    cur_iter_idx = 1
    rules = [int(str_rule) for str_rule in args.rule_series]
    agg_agents = []
    agg_win_batches = defaultdict(dict)
    agg_loss_batches = defaultdict(dict)

    for epoch in range(args.train_epochs):
        for rule_idx in rules:

            # if epoch % args.eval_factor == 0:
            #     for eval_rule_idx in rules:
            #         sp_agent = agent_dict[eval_rule_idx]["sp_agent"]
            #         bc_agent = agent_dict[eval_rule_idx]["bc_agent"]
            #         game = MultiTaskGame(
            #             sp_agent, bc_agent, cur_iter_idx, args, working_rule_dir
            #         )
            #         game.evaluate_lifelong_rules(cur_iter_idx, [eval_rule_idx], "train")

            sp_agent = agent_dict[rule_idx]["sp_agent"]
            bc_agent = agent_dict[rule_idx]["bc_agent"]

            print("Current rule: {}".format(rule_idx))
            game = MultiTaskGame(
                sp_agent, bc_agent, cur_iter_idx, args, working_rule_dir
            )

            rule = game.train_permute[rule_idx]
            print("Current rule: {}".format(rule))
            game.init_rule_games(rule)
            agent1, agent2 = game.start()

            agent1.train()
            agent2.train()

            pbar = tqdm(total=(args.num_sp * 2 + args.num_rb))

            while not game.finished():

                data = game.get_input()

                if len(data) == 0:
                    continue
                for key in data:
                    # print(key)
                    batch = to_device(data[key], device)

                    if key == "act1":
                        batch["actor"] = "act1"
                        reply = agent1.simulate(cur_iter_idx, batch)
                        t_count = agent1.update_logs(cur_iter_idx, batch, reply)

                    elif key == "act2":
                        batch["actor"] = "act2"
                        reply = agent2.simulate(cur_iter_idx, batch)
                        t_count = agent2.update_logs(cur_iter_idx, batch, reply)

                    else:
                        assert False

                    game.set_reply(key, reply)
                    pbar.update(t_count)

            if not args.split_train:
                if args.train_mode == "coach":
                    agent1.train_coach(cur_iter_idx)
                elif args.train_mode == "executor":
                    agent1.train_executor(cur_iter_idx)
                elif args.train_mode == "both":
                    agent1.train_both(cur_iter_idx)
                else:
                    raise Exception("Invalid train mode.")
                game.print_logs(cur_iter_idx)
                game.terminate()
            else:

                if cur_iter_idx % len(rules):
                    win_batches, loss_batches = agent1.train_coach(cur_iter_idx)
                    agg_win_batches.update(win_batches)
                    agg_loss_batches.update(loss_batches)
                    agg_agents.append((agent1, agent2))
                    game.print_logs(cur_iter_idx)
                    game.terminate(keep_agents=True)

                else:
                    win_batches, loss_batches = agent1.train_coach(cur_iter_idx)

                    agg_win_batches.update(win_batches)
                    agg_loss_batches.update(loss_batches)

                    # Change shuffling
                    agent1.train_executor(
                        cur_iter_idx,
                        agg_win_batches=agg_win_batches,
                        agg_loss_batches=agg_loss_batches,
                    )

                    for agent1, agent2 in agg_agents:
                        agent1.reset()
                        agent2.reset()

                    game.print_logs(cur_iter_idx)
                    game.terminate()

                    del agg_loss_batches
                    del agg_win_batches

                    agg_win_batches = defaultdict(dict)
                    agg_loss_batches = defaultdict(dict)

                cur_iter_idx += 1
                wandb.run.summary[f"max_iterations"] = cur_iter_idx
                pbar.close()

        del game

    writer.close()


if __name__ == "__main__":
    global device
    args = parse_args()
    self_play(args)
