import argparse
import os
import sys
import pprint

from set_path import append_sys_path
append_sys_path()

import torch
import tube
from pytube import DataChannelManager
import minirts
import numpy as np
import pickle
from inst_dict import InstructionDict

from collections import defaultdict, Counter
from rnn_coach import ConvRnnCoach
from onehot_coach import ConvOneHotCoach
from rnn_generator import RnnGenerator
from itertools import groupby
from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device, ResultStat, Logger
from best_models import best_executors, best_coaches

device = torch.device('cuda:0')

# def load_model(coach_path, executor_path):
#     if 'onehot' in coach_path:
#         coach = ConvOneHotCoach.load(coach_path).to(device)
#     elif 'gen' in coach_path:
#         coach = RnnGenerator.load(coach_path).to(device)
#     else:
#         coach = ConvRnnCoach.load(coach_path).to(device)
#     coach.max_raw_chars = 200
#     executor = Executor.load(executor_path).to(device)
#     executor_wrapper = ExecutorWrapper(
#         coach, executor, coach.num_instructions, 200, 0, 'full')
#     executor_wrapper.train(False)
#     return executor_wrapper
#
#
# def get_category(inst):
#     categories = {
#         "Mine" : ["mine", "mineral", "mining"],
#         "Create" : ["build", "make", "create", "train"],
#         "Attack" : ["attack", "destroy", "kill"]}
#
#     for cat, l in categories.items():
#         for s in l:
#             if s in inst:
#                 return cat
#
#     return inst
#
#
# def get_strategy(trajectories):
#     macro_strat = []
#     for traj in trajectories:
#         inst_cat = []
#         for inst in traj:
#             cat = get_category(inst)
#             inst_cat.append(cat)
#         macro_strat.append(inst_cat)
#     return macro_strat

if __name__ == '__main__':

    kind = "-200-rnn500"
    num_agents = 150
    num_games = 20

    # kind = "-500"
    # num_agents = 150
    # num_games = 15

    # kind = ""
    # num_agents = 60
    # num_games = 15

    with open("/home/ubuntu/out/p1dict{}.txt".format(kind), "rb") as fp:
        p1dict = pickle.load(fp)

    with open("/home/ubuntu/out/p2dict{}.txt".format(kind), "rb") as fp:
        p2dict = pickle.load(fp)

    with open("/home/ubuntu/out/p1_win_dict{}.txt".format(kind), "rb") as fp:
        p1_win_dict = pickle.load(fp)

    with open("/home/ubuntu/out/p2_win_dict{}.txt".format(kind), "rb") as fp:
        p2_win_dict = pickle.load(fp)

    with open("/home/ubuntu/out/masks{}.txt".format(kind), "rb") as fp:
        masks = pickle.load(fp)

    with open("/home/ubuntu/out/coach_inst_dict{}.txt".format(kind), "rb") as fp:
        inst_dict = pickle.load(fp)

    inst_count = []
    inst_counter = Counter()
    agent_win_counter = Counter()
    for agent in range(num_agents):
        for game in range(num_games):

            game_id = str(agent) + "_" + str(game)
            inst_count.append(len(Counter(p2dict[game_id])))

            for i in p2dict[game_id]:
                inst_counter[i] += 1

            if p1_win_dict[game_id] == 1:
                agent_win_counter[agent] += p1_win_dict[game_id]

        print("Agent {} win% = {}".format(agent, agent_win_counter[agent]*100/num_games))

    mean = np.mean(inst_count)
    std = np.std(inst_count)
    print("Mean: {}, std: {} for top{} model".format(mean, std, kind))

    instructions = ["attack", "build a workshop", "attack enemy", "build another dragon", "attack with dragon", "kill peasants",
                    "send idle peasant to mine", "attack peasant", "keep attacking", "kill peasant", "attack the enemy",
                    "attack enemy base", "build a stable", "make another dragon", "build a dragon", "kill archer", "make another archer",
                    "make a dragon", "build 2 dragon", "attack with archer"]
    winning_traj_mask = []

    for inst in instructions:
        winning_traj_mask.append(inst_dict.get_inst_idx(inst))

    winning_traj = []
    winning_traj_uniq = []
    losing_traj = []
    losing_traj_uniq = []

    for g_id, win in p1_win_dict.items():

        if p2_win_dict[g_id] == -1 and p1_win_dict[g_id] == -1:
            continue

        if win == 1:
            # Add winning trajectories
            win_traj = [i[0] for i in groupby(p1dict[g_id])]
            winning_traj_uniq.append(win_traj)
            winning_traj.append(p1dict[g_id])

            # Add losing trajectories
            loss_traj = [i[0] for i in groupby(p2dict[g_id])]
            losing_traj_uniq.append(loss_traj)
            losing_traj.append(p2dict[g_id])

        elif win == -1:
            assert p2_win_dict[g_id] == 1

            win_traj = [i[0] for i in groupby(p2dict[g_id])]
            winning_traj_uniq.append(win_traj)
            winning_traj.append(p2dict[g_id])

            # Add losing trajectories
            loss_traj = [i[0] for i in groupby(p1dict[g_id])]
            losing_traj_uniq.append(loss_traj)
            losing_traj.append(p1dict[g_id])

    winning_counter = Counter()
    losing_counter = Counter()

    for game in winning_traj:
        for inst in game:
            winning_counter[inst] += 1


    for game in losing_traj:
        for inst in game:
            losing_counter[inst] += 1

    inst_win_counter = Counter()
    inst_loss_counter = Counter()

    for inst, count in winning_counter.most_common():
        for game in winning_traj:
            if inst in game:
                inst_win_counter[inst]+=1

    for inst, count in losing_counter.most_common():
        for game in losing_traj:
            if inst in game:
                inst_loss_counter[inst]+=1

    top_100_win_counter = set([x for x, y in winning_counter.most_common()[0:100]])
    top_100_losing_counter = set([x for x, y in losing_counter.most_common()[0:100]])

    top_100_inst_win_counter = set([x for x, y in inst_win_counter.most_common()[0:100]])
    top_100_inst_loss_counter = set([x for x, y in inst_loss_counter.most_common()[0:100]])
    #
    # # winning_traj_uniq_macro = get_strategy(winning_traj_uniq)
    # # winning_traj_macro = get_strategy(winning_traj)
    # # losing_traj_macro = get_strategy(losing_traj)
    # # losing_traj_uniq_macro = get_strategy(losing_traj_uniq)
    #
    # coach = best_coaches['rnn50']
    # executor = best_executors['rnn']
    #
    # model = load_model(coach, executor)
    print("Terminated. ")
