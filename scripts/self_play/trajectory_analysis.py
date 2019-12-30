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
import pickle

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

def load_model(coach_path, executor_path):
    if 'onehot' in coach_path:
        coach = ConvOneHotCoach.load(coach_path).to(device)
    elif 'gen' in coach_path:
        coach = RnnGenerator.load(coach_path).to(device)
    else:
        coach = ConvRnnCoach.load(coach_path).to(device)
    coach.max_raw_chars = 200
    executor = Executor.load(executor_path).to(device)
    executor_wrapper = ExecutorWrapper(
        coach, executor, coach.num_instructions, 200, 0, 'full')
    executor_wrapper.train(False)
    return executor_wrapper


def get_category(inst):
    categories = {
        "Mine" : ["mine", "mineral", "mining"],
        "Create" : ["build", "make", "create", "train"],
        "Attack" : ["attack", "destroy", "kill"]}

    for cat, l in categories.items():
        for s in l:
            if s in inst:
                return cat

    return inst


def get_strategy(trajectories):
    macro_strat = []
    for traj in trajectories:
        inst_cat = []
        for inst in traj:
            cat = get_category(inst)
            inst_cat.append(cat)
        macro_strat.append(inst_cat)
    return macro_strat

if __name__ == '__main__':

    with open("/afs/csail.mit.edu/u/a/apjacob/winning_traj_uniq.txt", "rb") as fp:
        winning_traj_uniq = pickle.load(fp)

    with open("/afs/csail.mit.edu/u/a/apjacob/winning_traj.txt", "rb") as fp:
        winning_traj = pickle.load(fp)

    with open("/afs/csail.mit.edu/u/a/apjacob/losing_traj.txt", "rb") as fp:
        losing_traj = pickle.load(fp)

    with open("/afs/csail.mit.edu/u/a/apjacob/losing_traj_uniq.txt", "rb") as fp:
        losing_traj_uniq = pickle.load(fp)

    winning_counter = Counter()
    losing_counter = Counter()

    limit = 50

    for game in winning_traj:
        count = 0
        for inst in game:
            count += 1
            if count == limit:
                break

            winning_counter[inst] += 1


    for game in losing_traj:
        count = 0
        for inst in game:
            count += 1
            if count == limit:
                break

            losing_counter[inst] += 1

    # winning_traj_uniq_macro = get_strategy(winning_traj_uniq)
    # winning_traj_macro = get_strategy(winning_traj)
    # losing_traj_macro = get_strategy(losing_traj)
    # losing_traj_uniq_macro = get_strategy(losing_traj_uniq)

    coach = best_coaches['rnn50']
    executor = best_executors['rnn']

    model = load_model(coach, executor)
    print("Terminated. ")
