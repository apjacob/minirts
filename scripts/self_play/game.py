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
from game_utils import *
from agent import Agent
from jinja2 import Environment, FileSystemLoader
from common_utils import to_device, ResultStat, Logger
reward_tuple = [("win", 1), ("loss", -1)]
import wandb
import yaml
from tqdm import tqdm
import os
import shutil

UNITS = ['swordman', 'spearman', 'cavalry', 'archer', 'dragon']
UNIT_DICT = {unit: i for i, unit in enumerate(UNITS)}

class Game:
    def __init__(self, sp_agent, bc_agent, index, args):
        self.sp_agent = sp_agent
        self.bc_agent = bc_agent
        self.tb_log = args.tb_log
        self.args = args
        self.index = index
        self.dc = None
        self.context = None
        self.act1_dc = None
        self.act2_dc = None
        self.game_iter = 0
        self.agent1 = sp_agent
        self.agent2 = bc_agent

    def init_games(self):
        game_option = get_game_option(self.args)
        ai1_option, ai2_option = get_ai_options(
            self.args, [self.agent1.model.coach.num_instructions, self.agent1.model.coach.num_instructions])

        ## Launching games
        # self.context, self.act1_dc, self.act2_dc = init_games(
        #     self.args.num_thread, ai1_option, ai2_option, game_option)

        if self.args.opponent == "sp":
            self.context, self.act1_dc, self.act2_dc = init_mt_games(self.args.num_thread, 0, self.args, ai1_option, ai2_option,
                                                                 game_option)
        else:
            self.context, self.act1_dc, self.act2_dc = init_mt_games(0, self.args.num_thread, self.args, ai1_option,
                                                                     ai2_option,
                                                                     game_option)

        if self.index % self.args.sp_factor == 0:
            print("Playing against itself")
            self.agent1 = self.sp_agent
            self.agent2 = self.sp_agent.clone(model_type="current")
        else:
            print("Playing against BC model")
            self.agent1 = self.sp_agent
            self.agent2 = self.bc_agent

    def finished(self):
        return self.context.terminated()

    def terminate(self):
        self.dc.terminate()

        self.agent1.reset()
        self.agent2.reset()

    def print_logs(self, index, split='Train'):

        result1 = self.agent1.result
        result2 = self.agent2.result
        print(result1.log(0))
        print(result2.log(0))

        a1_win = result1.win / result1.num_games
        a1_loss = result1.loss / result1.num_games

        if self.args.opponent == "rb":
            a2_win = result1.loss / result1.num_games
            a2_loss = result1.win / result1.num_games
        else:
            a2_win = result2.win / result2.num_games
            a2_loss = result2.loss / result2.num_games


        if self.tb_log:
            wandb.log({'{}/Agent-1/Win'.format(split): a1_win,
                       '{}/Agent-1/Loss'.format(split): a1_loss,
                       '{}/Agent-2/Win'.format(split): a2_win,
                       '{}/Agent-2/Loss'.format(split): a2_loss}, step=index)

            self.agent1.tb_writer.add_scalar('{}/Agent-1/Win'.format(split), a1_win, index)
            self.agent1.tb_writer.add_scalar('{}/Agent-1/Loss'.format(split), a1_loss, index)

            self.agent2.tb_writer.add_scalar('{}/Agent-2/Win'.format(split), a2_win, index)
            self.agent2.tb_writer.add_scalar('{}/Agent-2/Loss'.format(split), a2_loss, index)

    def set_reply(self, key, reply):
        assert self.dc is not None

        return self.dc.set_reply(key, reply)

    def get_input(self):
        assert self.dc is not None
        self.game_iter += 1

        return self.dc.get_input(max_timeout_s=1)

    def start(self):
        assert self.context is not None

        self.context.start()
        self.dc = DataChannelManager([self.act1_dc, self.act2_dc])

        return self.agent1, self.agent2


class MultiTaskGame(Game):
    def __init__(self, agent1, agent2, index, args, working_rule_dir):
        super().__init__(agent1, agent2, index, args)

        attack_multipliers = os.path.join(self.args.rule_dir, "attack_multiplier.yaml")
        with open(attack_multipliers) as f:
            self.attack_multipliers = yaml.load(f, Loader=yaml.FullLoader)

        train_permute = os.path.join(self.args.rule_dir, "train_permute.yaml")
        with open(train_permute) as f:
            self.train_permute = yaml.load(f, Loader=yaml.FullLoader)

        self.num_train_rules = len(self.train_permute)

        valid_permute = os.path.join(self.args.rule_dir, "valid_permute.yaml")
        with open(valid_permute) as f:
            self.valid_permute = yaml.load(f, Loader=yaml.FullLoader)

        self.num_valid_rules = len(self.valid_permute)

        test_permute = os.path.join(self.args.rule_dir, "test_permute.yaml")
        with open(test_permute) as f:
            self.test_permute = yaml.load(f, Loader=yaml.FullLoader)

        self.num_test_rules = len(self.test_permute)
        self.working_rule_dir = working_rule_dir


    def init_rule_games(self, rule, num_sp=None, num_rb=None):
        lua_files = self.generate_files(rule)

        os.environ['LUA_PATH'] = os.path.join(lua_files, '?.lua')
        print('lua path:', os.environ['LUA_PATH'])

        game_option = get_game_option(self.args, lua_files)
        ai1_option, ai2_option = get_ai_options(
            self.args, [self.agent1.model.coach.num_instructions, self.agent1.model.coach.num_instructions])

        ## Launching games
        if num_sp is None or num_rb is None:
            num_sp = self.args.num_sp
            num_rb = self.args.num_rb

        self.context, self.act1_dc, self.act2_dc = init_mt_games(num_sp, num_rb, self.args, ai1_option, ai2_option,
                                                                 game_option)


    def generate_files(self, rule):
        file_loader = FileSystemLoader(self.working_rule_dir)
        env = Environment(loader=file_loader)
        unit_factory_template = env.get_template('unit_factory_template.txt')

        #default order: swordman, spearman, cavalry, archer, dragon
        rule_dict = {"swordman": self.attack_multipliers[rule[0]]['attack_multiplier'],
                     "spearman": self.attack_multipliers[rule[1]]['attack_multiplier'],
                     "cavalry": self.attack_multipliers[rule[2]]['attack_multiplier'],
                     "archer": self.attack_multipliers[rule[3]]['attack_multiplier'],
                     "dragon": self.attack_multipliers[rule[4]]['attack_multiplier'],}

        output = unit_factory_template.render(data=rule_dict)

        with open(os.path.join(self.working_rule_dir, "unit_factory.lua"), "w") as fh:
            fh.write(output)

        return self.working_rule_dir

    def evaluate(self, epoch, split='valid', num_rules=5):
        device = torch.device('cuda:%d' % self.args.gpu)
        num_games = 100

        if split=='valid':
            permute = self.valid_permute
        elif split=='test':
            permute = self.test_permute
        elif split=='train':
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_idx in range(num_rules): ##TODO: Not randomized
            rule = permute[rule_idx]
            self.init_rule_games(rule, num_sp=0, num_rb=num_games)
            agent1, agent2  = self.start()

            agent1.eval()
            agent2.eval()

            pbar = tqdm(total=num_games)

            while not self.finished():

                data = self.get_input()

                if len(data) == 0:
                    continue
                for key in data:
                    # print(key)
                    batch = to_device(data[key], device)

                    if key == 'act1':
                        batch['actor'] = 'act1'
                        reply = agent1.simulate(cur_iter_idx, batch)
                        t_count = agent1.update_logs(cur_iter_idx, batch, reply)

                    elif key == 'act2':
                        batch['actor'] = 'act2'
                        reply = agent2.simulate(cur_iter_idx, batch)
                        t_count = agent2.update_logs(cur_iter_idx, batch, reply)

                    else:
                        assert False

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_idx] = {"win": a1_result.win / a1_result.num_games,
                                 "loss": a1_result.loss / a1_result.num_games}

            cur_iter_idx += 1
            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log({"{}/{}/Win".format(split, rule): wl["win"],
                       "{}/{}/Loss".format(split, rule): wl["loss"]}, step=epoch)
            avg_win_rate += wl["win"]/num_rules

        print("Average win rate: {}".format(avg_win_rate))
        if avg_win_rate > self.agent1.best_test_win_pct:
            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return results

    def evaluate_rules(self, epoch, rule_idx, split='valid'):
        device = torch.device('cuda:%d' % self.args.gpu)
        num_games = 100

        if split=='valid':
            permute = self.valid_permute
        elif split=='test':
            permute = self.test_permute
        elif split=='train':
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_id in rule_idx: ##TODO: Not randomized
            rule = permute[rule_id]
            self.init_rule_games(rule, num_sp=0, num_rb=num_games)
            agent1, agent2  = self.start()

            agent1.eval()
            agent2.eval()

            pbar = tqdm(total=num_games)

            while not self.finished():

                data = self.get_input()

                if len(data) == 0:
                    continue
                for key in data:
                    # print(key)
                    batch = to_device(data[key], device)

                    if key == 'act1':
                        batch['actor'] = 'act1'
                        reply = agent1.simulate(cur_iter_idx, batch)
                        t_count = agent1.update_logs(cur_iter_idx, batch, reply)

                    elif key == 'act2':
                        batch['actor'] = 'act2'
                        reply = agent2.simulate(cur_iter_idx, batch)
                        t_count = agent2.update_logs(cur_iter_idx, batch, reply)

                    else:
                        assert False

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_id] = {"win": a1_result.win / a1_result.num_games,
                                 "loss": a1_result.loss / a1_result.num_games}

            cur_iter_idx += 1
            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log({"{}/{}/Win".format(split, rule): wl["win"],
                       "{}/{}/Loss".format(split, rule): wl["loss"]})
            avg_win_rate += wl["win"]/len(rule_idx)

        print("Average win rate: {}".format(avg_win_rate))
        if avg_win_rate > self.agent1.best_test_win_pct:
            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return results

def create_working_dir(args, working_dir):

    if os.path.exists(working_dir):
        print("Attempting to create an existing folder..")
        import pdb
        pdb.set_trace()

    os.makedirs(working_dir)
    src = args.rule_dir
    dest = working_dir
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

    print("Created working rule directory at: {}".format(dest))