import argparse
import os
import sys
import pprint
from set_path import append_sys_path
append_sys_path()
import torch
import random
import tube
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
import copy

reward_tuple = [("win", 1), ("loss", -1)]


class Agent:

    def __init__(self, coach, executor, device, args, writer, trainable=False, exec_sample=False):

        self.__coach        = coach
        self.__executor     = executor
        self.__trainable    = trainable
        self.device         = device
        self.args           = args
        self.tb_log         = args.tb_log
        self.save_folder    = None
        self.best_test_win_pct = 0
        self.tb_writer      = writer
        self.traj_dict      = defaultdict(list)
        self.result_dict    = {}
        self.result         = ResultStat('reward', None)
        self.model = self.load_model(self.__coach, self.__executor, self.args)
        self.exec_sample    = exec_sample

        if self.__trainable:
            self.optimizer = optim.Adam(self.model.coach.parameters(), lr=args.lr)
            self.sa_buffer = StateActionBuffer(max_buffer_size=args.max_table_size,
                                          buffer_add_prob=args.sampling_freq)
        else:
            self.optimizer = None
            self.sa_buffer = None

    def clone(self, model_type='current'):
        if model_type == 'current':
            print("Cloning current model.")
            cpy = Agent(self.__coach, self.__executor, self.device, self.args, self.tb_writer, trainable=False)
            cpy.model = self.model
        else:
            raise NameError

        return cpy
    def load_model(self, coach_path, model_path, args):
        if 'onehot' in coach_path:
            coach = ConvOneHotCoach.load(coach_path).to(self.device)
        elif 'gen' in coach_path:
            coach = RnnGenerator.load(coach_path).to(self.device)
        else:
            coach = ConvRnnCoach.load(coach_path).to(self.device)
        coach.max_raw_chars = args.max_raw_chars
        executor = Executor.load(model_path).to(self.device)
        executor_wrapper = ExecutorWrapper(
            coach, executor, coach.num_instructions, args.max_raw_chars, args.cheat, args.inst_mode)
        executor_wrapper.train(False)
        return executor_wrapper

    def simulate(self, index, batch):
        with torch.no_grad():
            reply, coach_reply = self.model.forward(batch, self.exec_sample)

            if self.__trainable:
                assert self.sa_buffer is not None

                ## Add coach reply to state-reply dict
                # batch.update(coach_reply['samples'])
                rv = self.sa_buffer.push(index, batch)

            return reply

    def reset(self):
        if self.__trainable:
            assert len(self.sa_buffer) == 0


        self.result        = ResultStat('reward', None)
        self.traj_dict     = defaultdict(list)
        self.result_dict   = {}

    def update_logs(self, index, batch, reply):
        self.result.feed(batch)

        inst = reply['inst']  ## Get get_inst
        game_ids = batch['game_id'].cpu().numpy()
        terminals = batch['terminal'].cpu().numpy().flatten()
        rewards = batch['reward'].cpu().numpy().flatten()

        # print(len(state_table))
        games_terminated = 0
        for i, g_id in enumerate(game_ids):

            act_dict = self.traj_dict
            act_win_dict = self.result_dict
            dict_key = str(index) + "_" + str(g_id[0])

            act_dict[dict_key].append(self.model.coach.inst_dict.get_inst(inst[i]))

            if terminals[i] == 1:
                games_terminated+=1

                # print("Game {} has terminated.".format(g_id[0]))
                if rewards[i] == 1:
                    act_win_dict[dict_key] = 1
                elif rewards[i] == -1:
                    act_win_dict[dict_key] = -1

        return games_terminated

    def train(self):
        self.model.train()

    def eval_model(self, gen_id, other_agent):
        print("Evaluating model....")
        e_result1, e_result2 = run_eval(self.args, self.model, other_agent.model, self.device)
        test_win_pct = e_result1.win / e_result1.num_games

        if self.tb_log:
            self.tb_writer.add_scalar('Test/Agent-1/Win', e_result1.win / e_result1.num_games, gen_id)
            self.tb_writer.add_scalar('Test/Agent-1/Loss', e_result1.loss / e_result1.num_games, gen_id)

            self.tb_writer.add_scalar('Test/Eval_model/Win', e_result2.win / e_result2.num_games, gen_id)
            self.tb_writer.add_scalar('Test/Eval_model/Loss', e_result2.loss / e_result2.num_games, gen_id)

        if test_win_pct > self.best_test_win_pct:
            self.best_test_win_pct = test_win_pct
            self.save_coach(gen_id)

    def init_save_folder(self, log_name):

        self.save_folder = os.path.join(self.args.save_folder, log_name)

        if os.path.exists(self.save_folder):
            print("Attempting to create an existing folder..")
            import pdb
            pdb.set_trace()

        os.makedirs(self.save_folder)

    #######################
    ## Executor specific ##
    #######################

    def align_executor_actions(self, batches):
        # action_probs = ['glob_cont_prob',
        #  'cmd_type_prob',
        #  'gather_idx_prob',
        #  'attack_idx_prob',
        #  'unit_type_prob',
        #  'building_type_prob',
        #  'building_loc_prob',
        #  'move_loc_prob']

        action_samples = ['current_cmd_type',
                          'current_cmd_unit_type',
                          'current_cmd_x',
                          'current_cmd_y',
                          'current_cmd_gather_idx',
                          'current_cmd_attack_idx']

        action_samples_dict = {key: 't_' + key for key in action_samples}

        for (g_id, elements) in batches.items():

            for asp in action_samples_dict:
                elements[action_samples_dict[asp]] = elements[asp][1:]

            for (key, element) in elements.items():

                ## Alignment
                if key in action_samples_dict.values():
                    continue
                else:
                    elements[key] = element[:-1]


    def train_executor(self, gen_id):
        assert self.__trainable
        assert self.args.sampling_freq >= 1.0

        if len(self.sa_buffer):

            win_batches, loss_batches = self.sa_buffer.pop(self.result_dict)
            # self.align_executor_actions(win_batches)
            # self.align_executor_actions(loss_batches)

            l1_loss, mse_loss, value = self.__update_executor(win_batches, loss_batches)

            if self.tb_log:
                self.tb_writer.add_scalar('Loss/RL-Loss', l1_loss, gen_id)
                self.tb_writer.add_scalar('Loss/MSE-Loss', mse_loss, gen_id)
                self.tb_writer.add_scalar('Value', value, gen_id)
        else:
            print("State-Action Buffer is empty.")

    def __update_executor(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss(reduction='none')

        denom = np.sum([y['inst'].size()[0] for x, y in win_batches.items()]) + \
                np.sum([y['inst'].size()[0] for x, y in loss_batches.items()])

        l1_losses = 0
        mse_losses = 0
        total_values = 0

        for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):

            for game_id, batch in batches.items():

                episode_len = batch['reward'].size(0)
                intervals = list(range(0, episode_len, self.args.train_batch_size))
                interval_tuples_iter = zip(intervals, intervals[1:] + [episode_len])

                for (s, e) in interval_tuples_iter:

                    sliced_batch = slice_batch(batch, s, e)

                    log_prob, all_losses, value = self.model.get_executor_rl_train_loss(sliced_batch)
                    l1 = r*log_prob #(r * torch.ones_like(value) - value.detach()) * log_prob
                    l2 = mse(value, r * torch.ones_like(value))

                    l1_mean = -1.0 * l1.sum() / denom
                    mse_loss = 1.0 * l2.sum() / denom
                    policy_loss = l1_mean #+ mse_loss

                    policy_loss.backward()

                    l1_losses += l1_mean.item()
                    mse_losses += mse_loss.item()
                    total_values += value.sum().item()/denom


        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model.eval()
        return l1_losses, mse_losses, total_values

    def save_executor(self, index):
        pass

        assert self.save_folder is not None

        model_file = os.path.join(self.save_folder, 'best_coach_checkpoint_%d.pt' % index)
        print('Saving model coach to: ', model_file)
        self.model.coach.save(model_file)

    ####################
    ## Coach specific ##
    ####################

    def train_coach(self, gen_id):
        assert self.__trainable

        if len(self.sa_buffer):
            win_batches, loss_batches = self.sa_buffer.pop(self.result_dict)
            l1_loss, mse_loss, value = self.__update_coach(win_batches, loss_batches)

            if self.tb_log:
                self.tb_writer.add_scalar('Loss/RL-Loss', l1_loss, gen_id)
                self.tb_writer.add_scalar('Loss/MSE-Loss', mse_loss, gen_id)
                self.tb_writer.add_scalar('Value', value, gen_id)
        else:
            print("State-Action Buffer is empty.")

    def __update_coach(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss(reduction='none')

        denom = np.sum([y['inst'].size()[0] for x, y in win_batches.items()]) + \
                np.sum([y['inst'].size()[0] for x, y in loss_batches.items()])

        l1_loss_mean = 0
        mse_loss_mean = 0

        total_value = 0
        for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):
            # if kind == "win":
            #     denom = np.sum([y['inst'].size()[0] for x, y in win_batches.items()])
            # else:
            #     denom = np.sum([y['inst'].size()[0] for x, y in loss_batches.items()])

            for game_id, batch in batches.items():
                log_prob, value = self.model.get_coach_rl_train_loss(batch)
                l1 = (r * torch.ones_like(value) - value.detach()) * log_prob
                l2 = mse(value, r * torch.ones_like(value))

                l1_mean = -1.0 * (l1.sum()) / denom
                mse_loss = 0.1 * (l2.sum()) / denom
                policy_loss = l1_mean + mse_loss

                policy_loss.backward()
                l1_loss_mean += l1_mean.item()
                mse_loss_mean += mse_loss.item()

                total_value += value.sum().item() / denom

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        return l1_loss_mean, mse_loss_mean, total_value

    def save_coach(self, index):
        assert self.save_folder is not None

        model_file = os.path.join(self.save_folder, 'best_coach_checkpoint_%d.pt' % index)
        print('Saving model coach to: ', model_file)
        self.model.coach.save(model_file)


def run_eval(args, model1, model2, device):

    num_eval_games = 100
    pbar = tqdm(total=num_eval_games * 2)

    result1 = ResultStat('reward', None)
    result2 = ResultStat('reward', None)

    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_options(
        args, [model1.coach.num_instructions, model2.coach.num_instructions])

    context, act1_dc, act2_dc = init_games(
        num_eval_games, ai1_option, ai2_option, game_option)
    context.start()
    dc = DataChannelManager([act1_dc, act2_dc])

    i = 0
    model1.eval()
    model2.eval()

    while not context.terminated():
        i += 1
        # if i % 1000 == 0:
        #     print('%d, progress agent1: win %d, loss %d' % (i, result1.win, result1.loss))

        data = dc.get_input(max_timeout_s=1)
        if len(data) == 0:
            continue
        for key in data:
            # print(key)
            batch = to_device(data[key], device)
            if key == 'act1':
                batch['actor'] = 'act1'
                ## Add batches to state table using sampling before adding
                ## Add based on the game_id

                result1.feed(batch)
                with torch.no_grad():
                    reply, _ = model1.forward(batch, exec_sample=True)

            elif key == 'act2':
                batch['actor'] = 'act2'
                result2.feed(batch)

                with torch.no_grad():
                    reply, _ = model2.forward(batch)

            else:
                assert False

            dc.set_reply(key, reply)

            game_ids = batch['game_id'].cpu().numpy()
            terminals = batch['terminal'].cpu().numpy().flatten()

            for i, g_id in enumerate(game_ids):
                if terminals[i] == 1:
                    pbar.update(1)

    model1.train()
    model2.train()

    return result1, result2


def slice_batch(batch, s, e):
    sliced_batch = {}

    for k, v in batch.items():
        if k == "actor":
            sliced_batch[k] = v
        else:
            sliced_batch[k] = v[s:e]

    return sliced_batch

