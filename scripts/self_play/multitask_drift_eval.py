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
from agent import Agent, MultiExecutorAgent
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
from matplotlib import cm
from pop_utils import model_dicts
from datetime import datetime
from common_utils.global_consts import UNIT_TYPE_TO_IDX
import matplotlib.pyplot as plt
import seaborn as sns
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
    parser.add_argument("--max_tick", type=int, default=int(6e4))
    parser.add_argument("--no_terrain", action="store_true")
    parser.add_argument("--resource", type=int, default=500)
    parser.add_argument("--resource_dist", type=int, default=4)
    parser.add_argument("--fair", type=int, default=0)
    parser.add_argument("--save_replay_freq", type=int, default=0)
    parser.add_argument("--save_replay_per_games", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="matches2/dev")
    parser.add_argument(
        "--save_img_dir", type=str, default="/lingo/apjacob/minirts/imgs/"
    )

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
    wandb.run.name = f"multitask-pop-drift-eval-{wandb.run.id}-{date}-{args.tag}"
    # wandb.run.save()
    wandb.config.update(args)

    # print("args:")
    # pprint.pprint(vars(args))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    partial_json_save_dir = os.path.join(
        args.eval_folder, f"drift-eval-{date}-{args.tag}"
    )

    if os.path.exists(partial_json_save_dir):
        print("Attempting to create an existing folder.. hence skipping...")
    else:
        os.makedirs(partial_json_save_dir)

    coach1 = get_coach_path(model_dicts["ft_coach[7]"][0])
    executors = {
        "bc": get_executor_path(model_dicts["bc"][0]),
        "ft_both[80]": get_executor_path(model_dicts["ft_both[80]"][0]),
        "ft_pop[80,40,20]": get_executor_path(model_dicts["ft_pop[80,40,20]"][0]),
    }
    rules = [7]
    args.coach_random_init = True
    NUM_GAMES = 250

    args.coach1 = coach1
    args.coach2 = coach1
    args.executor2 = executors["bc"]
    _coach1 = os.path.basename(args.coach1).replace(".pt", "")
    _executor1 = os.path.basename(args.executor1).replace(".pt", "")

    log_name = "multitask-pop-analyze-drift_c1_type={}_c2_type={}__e1_type={}_e2_type={}__lr={}__num_sp={}__num_rb={}_{}_{}".format(
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
    sp_agent = MultiExecutorAgent(
        coach=args.coach1,
        executors=executors,
        device=device,
        args=args,
        writer=writer,
        trainable=False,
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

    print("Current rule: {}".format(rules[0]))
    sub_win_rates = []
    full_reply_dicts = []
    if not os.path.exists(os.path.join(args.save_img_dir, "matrices.npy")):
        for i in tqdm(range(NUM_GAMES)):
            game = MultiTaskGame(sp_agent, bc_agent, 0, args, working_rule_dir)
            # rule = randint(0, 80)
            result, reply_dicts = game.drift_analysis_games(
                0,
                rules,
                "train",
                viz=args.viz,
                num_games=1,
            )
            full_reply_dicts += reply_dicts
            game.terminate()
            del game

        inst_list = [
            sp_agent.model.coach.inst_dict._idx2inst[
                reply["bc_coach"]["inst"].squeeze().tolist()
            ]
            for reply in full_reply_dicts
        ]

        build_unit_list = []
        for inst in inst_list:
            if "create" in inst or "build" in inst or "train" in inst or "make" in inst:
                if "peasant" in inst:
                    unit = UNIT_TYPE_TO_IDX["PEASANT"]
                elif "dragon" in inst:
                    unit = UNIT_TYPE_TO_IDX["DRAGON"]
                elif "archer" in inst:
                    unit = UNIT_TYPE_TO_IDX["ARCHER"]
                elif "cavalry" in inst or "cavs" in inst:
                    unit = UNIT_TYPE_TO_IDX["CAVALRY"]
                elif "spearman" in inst:
                    unit = UNIT_TYPE_TO_IDX["SPEARMAN"]
                elif (
                    "swordman" in inst
                    or "swordsman" in inst
                    or "swords" in inst
                    or "sword" in inst
                ):
                    unit = UNIT_TYPE_TO_IDX["SWORDMAN"]
                elif "catapult" in inst:
                    unit = UNIT_TYPE_TO_IDX["CATAPULT"]
                else:
                    unit = None

                build_unit_list.append(unit)
            else:
                build_unit_list.append(None)

        build_building_list = []
        for inst in inst_list:
            if "create" in inst or "build" in inst or "train" in inst or "make" in inst:
                if "shop" in inst:
                    unit = UNIT_TYPE_TO_IDX["WORKSHOP"]
                elif "stable" in inst:
                    unit = UNIT_TYPE_TO_IDX["STABLE"]
                elif "barrack" in inst:
                    unit = UNIT_TYPE_TO_IDX["BARRACK"]
                elif "tower" in inst:
                    unit = UNIT_TYPE_TO_IDX["GUARD_TOWER"]
                elif "blacksmith" in inst:
                    unit = UNIT_TYPE_TO_IDX["BLACKSMITH"]
                else:
                    unit = None

                build_building_list.append(unit)
            else:
                build_building_list.append(None)

        bc_executor_unit_types = (
            np.asarray(
                [
                    (
                        reply["bc_executor"]["one_hot_reply"][
                            "unit_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["bc_executor"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [4, 6]]
                        #     .sum(1)
                        #     >= 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )
        ft_both_unit_type = (
            np.asarray(
                [
                    (
                        reply["ft_both[80]"]["one_hot_reply"][
                            "unit_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["ft_both[80]"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [4, 6]]
                        #     .sum(1)
                        #     >= 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )
        ft_pop_unit_type = (
            np.asarray(
                [
                    (
                        reply["ft_pop[80,40,20]"]["one_hot_reply"][
                            "unit_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["ft_pop[80,40,20]"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [4, 6]]
                        #     .sum(1)
                        #     >= 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )

        bc_executor_building_types = (
            np.asarray(
                [
                    (
                        reply["bc_executor"]["one_hot_reply"][
                            "building_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["bc_executor"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [3, 6]]
                        #     .sum(1)
                        #     > 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )
        ft_both_building_type = (
            np.asarray(
                [
                    (
                        reply["ft_both[80]"]["one_hot_reply"][
                            "building_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["ft_both[80]"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [3, 6]]
                        #     .sum(1)
                        #     > 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )
        ft_pop_building_type = (
            np.asarray(
                [
                    (
                        reply["ft_pop[80,40,20]"]["one_hot_reply"][
                            "building_type_prob"
                        ].squeeze()
                        # * (
                        #     reply["ft_pop[80,40,20]"]["one_hot_reply"]["cmd_type_prob"]
                        #     .squeeze()[:, [3, 6]]
                        #     .sum(1)
                        #     > 1.0
                        # ).unsqueeze(1)
                    )
                    .sum(0)
                    .tolist()[1:]
                    for reply in full_reply_dicts
                ]
            ).argmax(1)
            + 1
        )

        mat_bc_units = create_matrix(
            build_unit_list, bc_executor_unit_types, title="bc-bc"
        )
        mat_both_units = create_matrix(
            build_unit_list, ft_both_unit_type, title="bc-both"
        )
        mat_pop_units = create_matrix(build_unit_list, ft_pop_unit_type, title="bc-pop")

        mat_bc_buildings = create_matrix(
            build_building_list, bc_executor_building_types, title="bc-bc"
        )
        mat_both_buildings = create_matrix(
            build_building_list, ft_both_building_type, title="bc-both"
        )
        mat_pop_buildings = create_matrix(
            build_building_list, ft_pop_building_type, title="bc-pop"
        )

        mat_bc = mat_bc_units  # + mat_bc_buildings
        mat_both = mat_both_units  # + mat_both_buildings
        mat_pop = mat_pop_units  # + mat_pop_buildings

        print("Saving Numpy matrices...")
        with open(os.path.join(args.save_img_dir, "matrices.npy"), "wb") as f:
            np.save(f, mat_bc)
            np.save(f, mat_both)
            np.save(f, mat_pop)
    else:
        print("Loading Numpy matrices...")
        with open(os.path.join(args.save_img_dir, "matrices.npy"), "rb") as f:
            mat_bc = np.load(f)
            mat_both = np.load(f)
            mat_pop = np.load(f)

    plot_matrices(
        mat_bc, mat_both, mat_pop, title="Original", save_dir=args.save_img_dir
    )

    print_summary(mat_bc, mat_both, mat_pop, "original")

    diff_bc_bc = np.absolute(mat_bc - mat_bc)
    diff_both_bc = np.absolute(mat_both - mat_bc)
    diff_pop_bc = np.absolute(mat_pop - mat_bc)
    plot_matrices(
        diff_bc_bc,
        diff_both_bc,
        diff_pop_bc,
        title="Original-mat_bc",
        save_dir=args.save_img_dir,
    )

    print_summary(diff_bc_bc, diff_both_bc, diff_pop_bc, "bc")

    diff_bc_both = np.absolute(mat_bc - mat_both)
    diff_both_both = np.absolute(mat_both - mat_both)
    diff_pop_both = np.absolute(mat_pop - mat_both)

    plot_matrices(
        diff_bc_both,
        diff_both_both,
        diff_pop_both,
        title="Original-mat_both",
        save_dir=args.save_img_dir,
    )

    print_summary(diff_bc_both, diff_both_both, diff_pop_both, "both")

    diff_bc_pop = np.absolute(mat_bc - mat_pop)
    diff_both_pop = np.absolute(mat_both - mat_pop)
    diff_pop_pop = np.absolute(mat_pop - mat_pop)

    plot_matrices(
        diff_bc_pop,
        diff_both_pop,
        diff_pop_pop,
        title="Original-mat_pop",
        save_dir=args.save_img_dir,
    )

    print_summary(diff_bc_pop, diff_both_pop, diff_pop_pop, "pop")

    sub_win_rates.append(result[args.rule]["win"])
    writer.close()


def print_summary(A, B, C, tag):
    dia = np.diag_indices(12)
    print("#" * 50)
    print(
        f"Abs difference with {tag}: "
        f"bc: {A.sum()}, off_diag sum: {A.sum() - A[dia].sum()}\n"
        f"both: {B.sum()}, off_diag sum: {B.sum() - B[dia].sum()}\n"
        f"pop: {C.sum()}, off_diag sum: {C.sum() - C[dia].sum()}\n"
    )
    print("#" * 50)


def plot_matrices(m1, m2, m3, title="", save_dir="/lingo/apjacob/minirts/imgs/"):
    plt.figure(dpi=10000)
    plt.figure(figsize=(9, 16))
    f, (ax2, ax3, axcb) = plt.subplots(
        1, 3, gridspec_kw={"width_ratios": [1.5, 1.5, 0.08]}
    )
    ax2.get_shared_y_axes().join(ax3)
    dia = np.diag_indices(6)

    k1 = np.copy(m1)
    k2 = np.copy(m2)
    k3 = np.copy(m3)
    k1 = k1[0:6, 0:6]
    k2 = k2[0:6, 0:6]
    #
    k3 = k3[0:6, 0:6]
    mask = np.zeros_like(k1, dtype=bool)
    mask[dia] = False
    # k1 = k1 / k1.max()
    # k2 = k2 / k2.max()
    # k3 = k3 / k3.max()

    k1[dia] = np.nan
    k2[dia] = np.nan
    k3[dia] = np.nan
    # g1 = sns.heatmap(
    #     m1,
    #     cbar=False,
    #     ax=ax1,
    #     square=True,
    #     # vmax=1.0,
    #     # vmin=0.0,
    #     cmap="YlGnBu",
    #     linewidths=0.01,
    #     yticklabels=False,
    #     xticklabels=False,
    # )
    # g1.set_title("BC")
    # g1.set_ylabel("")
    # g1.set_xlabel("")
    blue = cm.get_cmap("Blues")
    blue.set_bad("grey")
    UNITS = ["Peasant", "Spearman", "Swordman", "Cavalry", "Dragon", "Archer"]
    ticks = [x + 0.5 for x in range(6)]
    g2 = sns.heatmap(
        k2,
        cbar=False,
        ax=ax2,
        square=True,
        vmax=0.4,
        vmin=0.0,
        robust=True,
        cmap=blue,
        yticklabels=False,
        linewidths=0.01,
        xticklabels=False,
    )
    g2.set_title(r"$RL^{joint}$", fontname="serif", fontweight="bold")
    g2.set_ylabel("Executor action", fontname="serif", fontweight="bold")
    g2.set_facecolor("grey")
    # g2.set_xlabel("")
    g2.set_xticks(ticks)  # <--- set the ticks first
    g2.set_xticklabels(UNITS)
    g2.set_yticks(ticks)  # <--- set the ticks first
    g2.set_yticklabels(UNITS)

    g3 = sns.heatmap(
        k3,
        ax=ax3,
        cbar_ax=axcb,
        square=True,
        cmap=blue,
        robust=True,
        vmax=0.4,
        vmin=0.0,
        linewidths=0.01,
        yticklabels=False,
        xticklabels=False,
    )
    g3.set_title(r"$RL^{multi}$", fontweight="bold")
    # g3.set_ylabel("")
    # g3.set_xlabel("")
    g3.set_xticks(ticks)  # <--- set the ticks first
    g3.set_xticklabels(UNITS)
    g3.set_facecolor("grey")

    f.text(
        0.5,
        0.04,
        "Instructor message",
        ha="center",
        fontname="serif",
        fontweight="bold",
    )
    # plt.suptitle(title)
    # may be needed to rotate the ticklabels correctly:
    for ax in [g2, g3]:
        tl = ax.get_xticklabels()
        for tick in ax.get_xticklabels():
            tick.set_fontname("serif")
            # tick.set_fontweight("bold")
        ax.set_xticklabels(tl, rotation=45)
        tly = ax.get_yticklabels()
        for tick in ax.get_yticklabels():
            tick.set_fontname("serif")
            # tick.set_fontweight("bold")
        ax.set_yticklabels(tly, rotation=0)

    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.jpg"), bbox_inches="tight")
    plt.show()
    plt.clf()


def create_matrix(build_unit_list, low_level_types, title="default"):
    filtered_zip_list = zip(build_unit_list, low_level_types)
    filtered_zip_list = [(x, y) for x, y in filtered_zip_list if x is not None]
    matrix = np.zeros([17, 17])

    for x, y in filtered_zip_list:
        matrix[x][y] += 1.0

    consts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
    matrix = matrix[consts, :][:, consts]
    matrix = np.nan_to_num(matrix / matrix.sum(0))
    dia = np.diag_indices(12)
    matrix[dia] = 0.0

    return matrix


if __name__ == "__main__":
    global device
    args = parse_args()
    self_play(args)
