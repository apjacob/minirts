import argparse
import os
import sys
import pprint

from set_path import append_sys_path
append_sys_path()

import torch
import random
import tube
import copy
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
from game_utils import *
from agent import Agent
reward_tuple = [("win", 1), ("loss", -1)]

class Game:
    def __init__(self, sp_agent, bc_agent, index, args):
        self.sp_agent = sp_agent
        self.bc_agent = bc_agent
        self.tb_log = args.tb_log
        self.args = args
        self.dc = None

        if index % self.args.sp_factor == 0:
            print("Playing against itself")
            self.agent1 = self.sp_agent
            self.agent2 = self.sp_agent.clone(model_type="current")
        else:
            print("Playing against BC model")
            self.agent1 = self.sp_agent
            self.agent2 = self.bc_agent

        self.game_iter = 0

        game_option = get_game_option(args)
        ai1_option, ai2_option = get_ai_options(
            args, [self.agent1.model.coach.num_instructions, self.agent1.model.coach.num_instructions])

        ## Launching games
        self.context, self.act1_dc, self.act2_dc = init_games(
            args.num_thread, ai1_option, ai2_option, game_option)

    def finished(self):
        return self.context.terminated()

    def terminate(self):
        self.dc.terminate()

        self.agent1.reset()
        self.agent2.reset()

    def print_logs(self, index):

        result1 = self.agent1.result
        result2 = self.agent2.result
        print(result1.log(0))
        print(result2.log(0))

        if self.tb_log:
            self.agent1.tb_writer.add_scalar('Train/Agent-1/Win', result1.win / result1.num_games, index)
            self.agent1.tb_writer.add_scalar('Train/Agent-1/Loss', result1.loss / result1.num_games, index)

            self.agent2.tb_writer.add_scalar('Train/Agent-2/Win', result2.win / result2.num_games, index)
            self.agent2.tb_writer.add_scalar('Train/Agent-2/Loss', result2.loss / result2.num_games, index)

    def set_reply(self, key, reply):
        assert self.dc is not None

        return self.dc.set_reply(key, reply)

    def get_input(self):
        assert self.dc is not None
        self.game_iter += 1

        return self.dc.get_input(max_timeout_s=1)

    def start(self):
        self.context.start()
        self.dc = DataChannelManager([self.act1_dc, self.act2_dc])

        return self.agent1, self.agent2

