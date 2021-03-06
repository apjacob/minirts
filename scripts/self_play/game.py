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
from collections import defaultdict, Counter
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

UNITS = ["swordman", "spearman", "cavalry", "archer", "dragon"]
UNIT_DICT = {unit: i for i, unit in enumerate(UNITS)}
# rps_dict = {
#     "swordman": "Effective against spearmen, No bonus against {archer, swordman}, Not effective against cavalry, Cannot attack dragons",
#     "spearman": "Effective against cavalry, No bonus against {archer, spearman}, Not effective against swordman, Cannot attack dragons",
#     "cavalry": "Effective against swordman, No bonus against {archer, cavalry}, Not effective against spearman, Cannot attack dragons",
#     "archer": "Effective against dragon, Not effective against {spearmen, cavalry, swordman, archer}",
#     "dragon": "No bonus against {spearmen, cavalry, swordman, dragon}, Not effective against archer",
# }

rps_dict = {
    "swordman": "1.0 & 1.5 & 0.5 & 1.0 & 0.0",
    "spearman": "0.5 & 1.0 & 1.5 & 1.0 & 0.0",
    "cavalry": "1.5 & 0.5 & 1.0 & 1.0 & 0.0",
    "archer": "0.5 & 0.5 & 0.5 & 0.5 & 2.0",
    "dragon": "1.0 & 1.0 & 1.0 & 0.5 & 1.0",
}


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
            self.args,
            [
                self.agent1.model.coach.num_instructions,
                self.agent1.model.coach.num_instructions,
            ],
        )

        ## Launching games
        # self.context, self.act1_dc, self.act2_dc = init_games(
        #     self.args.num_thread, ai1_option, ai2_option, game_option)

        if self.args.opponent == "sp":
            self.context, self.act1_dc, self.act2_dc = init_mt_games(
                self.args.num_thread, 0, self.args, ai1_option, ai2_option, game_option
            )
        else:
            self.context, self.act1_dc, self.act2_dc = init_mt_games(
                0, self.args.num_thread, self.args, ai1_option, ai2_option, game_option
            )

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

    def terminate(self, keep_agents=False):
        self.dc.terminate()

        if not keep_agents:
            self.agent1.reset()
            self.agent2.reset()

    def print_logs(self, index, split="Train"):

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
            wandb.log(
                {
                    "{}/Agent-1{}/Win".format(split, self.agent1.tag): a1_win,
                    "{}/Agent-1{}/Loss".format(split, self.agent1.tag): a1_loss,
                    "{}/Agent-2{}/Win".format(split, self.agent1.tag): a2_win,
                    "{}/Agent-2{}/Loss".format(split, self.agent1.tag): a2_loss,
                },
                step=index,
            )

            self.agent1.tb_writer.add_scalar(
                "{}/Agent-1{}/Win".format(split, self.agent1.tag), a1_win, index
            )
            self.agent1.tb_writer.add_scalar(
                "{}/Agent-1{}/Loss".format(split, self.agent1.tag), a1_loss, index
            )

            self.agent2.tb_writer.add_scalar(
                "{}/Agent-2{}/Win".format(split, self.agent1.tag), a2_win, index
            )
            self.agent2.tb_writer.add_scalar(
                "{}/Agent-2{}/Loss".format(split, self.agent1.tag), a2_loss, index
            )

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

    def init_rule_games(self, rule, num_sp=None, num_rb=None, viz=False):
        lua_files = self.generate_files(rule)

        os.environ["LUA_PATH"] = os.path.join(lua_files, "?.lua")
        print("lua path:", os.environ["LUA_PATH"])

        game_option = get_game_option(self.args, lua_files)
        ai1_option, ai2_option = get_ai_options(
            self.args,
            [
                self.agent1.model.coach.num_instructions,
                self.agent1.model.coach.num_instructions,
            ],
        )

        ## Launching games
        if num_sp is None or num_rb is None:
            num_sp = self.args.num_sp
            num_rb = self.args.num_rb

        self.context, self.act1_dc, self.act2_dc = init_mt_games(
            num_sp, num_rb, self.args, ai1_option, ai2_option, game_option, viz=viz
        )

    def init_drift_games(self, rule, num_sp=None, num_rb=None, viz=False):
        lua_files = self.generate_files(rule)

        os.environ["LUA_PATH"] = os.path.join(lua_files, "?.lua")
        print("lua path:", os.environ["LUA_PATH"])

        game_option = get_game_option(self.args, lua_files)
        ai1_option, ai2_option = get_ai_options(
            self.args,
            [
                self.agent1.model.coach.num_instructions,
                self.agent1.model.coach.num_instructions,
            ],
        )

        ## Launching games
        if num_sp is None or num_rb is None:
            num_sp = self.args.num_sp
            num_rb = self.args.num_rb

        self.context, self.act1_dc, self.act2_dc = create_drift_games(
            num_sp, num_rb, self.args, ai1_option, ai2_option, game_option, viz=viz
        )

    def init_rule_games_botvbot(self, bot1idx, bot2idx, rule, num_games, viz=False):
        lua_files = self.generate_files(rule)

        os.environ["LUA_PATH"] = os.path.join(lua_files, "?.lua")
        print("lua path:", os.environ["LUA_PATH"])

        game_option = get_game_option(self.args, lua_files)
        ai1_option, ai2_option = get_ai_options(
            self.args,
            [
                self.agent1.model.coach.num_instructions,
                self.agent1.model.coach.num_instructions,
            ],
        )

        self.context, self.act1_dc, self.act2_dc = init_botvbot(
            bot1idx=bot1idx,
            bot2idx=bot2idx,
            num_games=num_games,
            args=self.args,
            ai1_option=ai1_option,
            ai2_option=ai2_option,
            game_option=game_option,
            viz=viz,
        )

    def init_rule_games_vbot(self, botidx, rule, num_games, viz=False):
        lua_files = self.generate_files(rule)

        os.environ["LUA_PATH"] = os.path.join(lua_files, "?.lua")
        print("lua path:", os.environ["LUA_PATH"])

        game_option = get_game_option(self.args, lua_files)
        ai1_option, ai2_option = get_ai_options(
            self.args,
            [
                self.agent1.model.coach.num_instructions,
                self.agent1.model.coach.num_instructions,
            ],
        )

        self.context, self.act1_dc, self.act2_dc = init_vbot(
            botidx=botidx,
            num_games=num_games,
            args=self.args,
            ai1_option=ai1_option,
            ai2_option=ai2_option,
            game_option=game_option,
            viz=viz,
        )

    def generate_files(self, rule):
        file_loader = FileSystemLoader(self.working_rule_dir)
        env = Environment(loader=file_loader)
        unit_factory_template = env.get_template("unit_factory_template.txt")

        # default order: swordman, spearman, cavalry, archer, dragon
        rule_dict = {
            "swordman": self.attack_multipliers[rule[0]]["attack_multiplier"],
            "spearman": self.attack_multipliers[rule[1]]["attack_multiplier"],
            "cavalry": self.attack_multipliers[rule[2]]["attack_multiplier"],
            "archer": self.attack_multipliers[rule[3]]["attack_multiplier"],
            "dragon": self.attack_multipliers[rule[4]]["attack_multiplier"],
        }

        output = unit_factory_template.render(data=rule_dict)

        with open(os.path.join(self.working_rule_dir, "unit_factory.lua"), "w") as fh:
            fh.write(output)

        return self.working_rule_dir

    def evaluate(self, epoch, split="valid", num_rules=5):
        print("Validating...")
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = 100

        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_idx in range(num_rules):  ##TODO: Not randomized
            rule = permute[rule_idx]
            self.init_rule_games(rule, num_sp=0, num_rb=num_games)
            print(f"Validating on rule ({rule_idx}): {rule}")

            agent1, agent2 = self.start()

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

                    rule_tensor = (
                        torch.tensor([UNIT_DICT[unit] for unit in rule])
                        .to(device)
                        .repeat(batch["game_id"].size(0), 1)
                    )
                    batch["rule_tensor"] = rule_tensor

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

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_idx] = {
                "win": a1_result.win / a1_result.num_games,
                "loss": a1_result.loss / a1_result.num_games,
            }

            cur_iter_idx += 1
            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "Zero/{}/{}/Win".format(split, rule): wl["win"],
                    "Zero/{}/{}/Loss".format(split, rule): wl["loss"],
                },
                step=epoch,
            )
            avg_win_rate += wl["win"] / num_rules

        print("Average win rate: {}".format(avg_win_rate))
        if avg_win_rate > self.agent1.best_test_win_pct:
            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return results

    def analyze_rule_games(
        self, epoch, rule_idx, split="valid", viz=False, num_games=100, num_sp=0
    ):
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = num_games

        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_id in rule_idx:  ##TODO: Not randomized
            rule = permute[rule_id]
            self.init_rule_games(rule, num_sp=num_sp, num_rb=num_games, viz=viz)
            agent1, agent2 = self.start()

            agent1.eval()
            agent2.eval()

            if num_sp > 0:
                pbar = tqdm(total=num_games * 2 + num_sp)
            else:
                pbar = tqdm(total=num_games)

            while not self.finished():

                data = self.get_input()

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

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_id] = {
                "win": a1_result.win / a1_result.num_games,
                "loss": a1_result.loss / a1_result.num_games,
            }

            if num_sp > 0:
                print("#" * 50)
                print(f"Win: {a1_result.win / a1_result.num_games}")
                print(f"Loss: {self.agent2.result.win / self.agent2.result.num_games}")
                print(
                    f"Draw: {(a1_result.loss - self.agent2.result.win) / self.agent2.result.num_games}"
                )
                print("#" * 50)

            cur_iter_idx += 1
            print(results)
            # counter = Counter()
            # for game_id, insts in agent1.traj_dict.items():
            #     for inst in insts:
            #         counter[inst] += 1
            #
            # print("##### TOP N Instructions #####")
            # print(counter.most_common(10))
            # print("##############################")

            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "{}/{}/Win".format(split, rule): wl["win"],
                    "{}/{}/Loss".format(split, rule): wl["loss"],
                }
            )
            avg_win_rate += wl["win"] / len(rule_idx)

        if num_sp == 0:
            print("Average win rate: {}".format(avg_win_rate))

        return results

    def print_rule_desc(self, rule_idx, split="train"):
        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")
        rule = permute[rule_idx]
        rule_rps_dict = rps_dict.copy()
        for i, unit in enumerate(rule):
            rule_rps_dict[UNITS[i]] = rps_dict[unit]

        rule_mappings = {
            80: "original",
            40: "B",
            20: "C",
            21: "D",
            14: "E",
            7: "F",
            3: "G",
            12: "H",
            13: "J",
        }
        rule_letter = rule_mappings[rule_idx]
        # print(f"############- Rule: {rule_idx} -###################")
        print("\\begin{table*}[h!]")
        print("\centering")
        print("\\begin{tabular}{|l||c|c|c|c|c|}")
        print("\hline")
        print("& \multicolumn{5}{|c|}{\\textbf{Attack multiplier}} \\\\")
        print("\hline")
        print("\hline")
        print(
            "\\textbf{Unit name} & \\textbf{Swordman} & \\textbf{Spearman} & \\textbf{Cavalry} & \\textbf{Archer} & \\textbf{Dragon} \\\\"
        )
        print("\hline")
        for unit, multiplier in rule_rps_dict.items():
            print(f"\\textsc{{{unit}}} & {multiplier} \\\\")
            print("\hline")

        print("\end{tabular}\\\\")
        print(
            f"\caption{{\label{{Table:{rule_letter}}} Rule "
            + f"{rule_letter} attack modifier}}"
        )
        print("\end{table*}")
        # print("#" * 50)

    def analyze_rule_games_vbot(
        self, epoch, rule_idx, split="valid", viz=False, num_games=100
    ):
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = num_games

        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        unitidx = [0, 1, 2, 3, 4]
        botidx = random.choice(unitidx)
        counter = Counter()
        idx2utype = [
            "SWORDMAN",
            "SPEARMAN",
            "CAVALRY",
            "ARCHER",
            "DRAGON",
        ]

        rule = permute[rule_idx]
        rule_rps_dict = rps_dict.copy()
        for i, unit in enumerate(rule):
            rule_rps_dict[UNITS[i]] = rps_dict[unit]

        print("############RULE RPS###################")
        for unit, multiplier in rule_rps_dict.items():
            print(f"{unit}: {multiplier}")
        print("#######################################")

        print(f"Playing against bot {idx2utype[botidx]}")
        self.init_rule_games_vbot(
            botidx=botidx, rule=rule, num_games=num_games, viz=viz
        )
        agent1, agent2 = self.start()

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

                self.set_reply(key, reply)
                pbar.update(t_count)

        a1_result = self.agent1.result

        results[rule_idx] = {
            "win": a1_result.win / a1_result.num_games,
            "loss": a1_result.loss / a1_result.num_games,
        }

        cur_iter_idx += 1
        print(results)

        for game_id, insts in agent1.traj_dict.items():
            for inst in insts:
                counter[inst] += 1

        pbar.close()
        self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "{}/{}/Win".format(split, rule): wl["win"],
                    "{}/{}/Loss".format(split, rule): wl["loss"],
                }
            )
            avg_win_rate += wl["win"]
        print(f"Top-10 Instructions: {counter.most_common(10)}")
        print(f"Average win rate: {avg_win_rate}")
        if avg_win_rate > self.agent1.best_test_win_pct:
            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return idx2utype[botidx], rule_idx, avg_win_rate, counter.most_common(10)

    def evaluate_rules(self, epoch, rule_idx, split="valid"):
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = 100

        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_id in rule_idx:  ##TODO: Not randomized
            rule = permute[rule_id]
            self.init_rule_games(rule, num_sp=0, num_rb=num_games)
            agent1, agent2 = self.start()

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

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_id] = {
                "win": a1_result.win / a1_result.num_games,
                "loss": a1_result.loss / a1_result.num_games,
            }

            cur_iter_idx += 1
            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "{}/{}/Win".format(split, rule): wl["win"],
                    "{}/{}/Loss".format(split, rule): wl["loss"],
                }
            )
            avg_win_rate += wl["win"] / len(rule_idx)

        print("Average win rate: {}".format(avg_win_rate))
        if avg_win_rate > self.agent1.best_test_win_pct:
            wandb.run.summary[f"best_win_rate{self.agent1.tag}"] = avg_win_rate
            wandb.run.summary[f"best_iteration{self.agent1.tag}"] = epoch

            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return results

    def evaluate_lifelong_rules(self, epoch, rule_series, split="train"):
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = 100

        if split == "valid":
            permute = self.valid_permute
        elif split == "test":
            permute = self.test_permute
        elif split == "train":
            permute = self.train_permute
        else:
            raise Exception("Invalid split.")

        cur_iter_idx = 0
        results = {}
        for rule_id in rule_series:
            rule = permute[rule_id]
            print("Evaluating current rule: {}".format(rule))
            self.init_rule_games(rule, num_sp=0, num_rb=num_games)
            agent1, agent2 = self.start()

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

                    self.set_reply(key, reply)
                    pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_id] = {
                "win": a1_result.win / a1_result.num_games,
                "loss": a1_result.loss / a1_result.num_games,
            }

            cur_iter_idx += 1
            pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "Lifelong/{}/{}/Win".format(split, rule): wl["win"],
                    "Lifelong/{}/{}/Loss".format(split, rule): wl["loss"],
                }
            )
            avg_win_rate += wl["win"] / len(rule_series)

        print("Average win rate: {}".format(avg_win_rate))
        if avg_win_rate > self.agent1.best_test_win_pct:
            wandb.run.summary[f"best_win_rate{self.agent1.tag}"] = avg_win_rate
            wandb.run.summary[f"best_iteration{self.agent1.tag}"] = epoch

            self.agent1.best_test_win_pct = avg_win_rate
            self.agent1.save_coach(epoch)
            self.agent1.save_executor(epoch)

        return results

    def drift_analysis_games(
        self, epoch, rule_idx, split="valid", viz=False, num_games=1
    ):
        device = torch.device("cuda:%d" % self.args.gpu)
        num_games = num_games
        permute = self.train_permute

        cur_iter_idx = 0
        results = {}
        reply_dicts = []
        for rule_id in rule_idx:  ##TODO: Not randomized
            rule = permute[rule_id]
            self.init_drift_games(rule, num_sp=0, num_rb=num_games, viz=viz)
            agent1, agent2 = self.start()

            agent1.eval()
            agent2.eval()

            # pbar = tqdm(total=num_games)

            while not self.finished():

                data = self.get_input()

                if len(data) == 0:
                    continue
                for key in data:
                    # print(key)
                    batch = to_device(data[key], device)

                    if key == "act1":
                        batch["actor"] = "act1"
                        reply, replies = agent1.simulate(cur_iter_idx, batch)
                        t_count = agent1.update_logs(cur_iter_idx, batch, reply)
                        reply_dicts.append(replies)

                    elif key == "act2":
                        batch["actor"] = "act2"
                        reply = agent2.simulate(cur_iter_idx, batch)
                        t_count = agent2.update_logs(cur_iter_idx, batch, reply)

                    else:
                        assert False

                    self.set_reply(key, reply)
                    # pbar.update(t_count)

            a1_result = self.agent1.result

            results[rule_id] = {
                "win": a1_result.win / a1_result.num_games,
                "loss": a1_result.loss / a1_result.num_games,
            }

            cur_iter_idx += 1
            print(results)
            # counter = Counter()
            # for game_id, insts in agent1.traj_dict.items():
            #     for inst in insts:
            #         counter[inst] += 1
            #
            # print("##### TOP N Instructions #####")
            # print(counter.most_common(10))
            # print("##############################")

            # pbar.close()
            self.terminate()

        avg_win_rate = 0
        for rule, wl in results.items():
            wandb.log(
                {
                    "{}/{}/Win".format(split, rule): wl["win"],
                    "{}/{}/Loss".format(split, rule): wl["loss"],
                }
            )
            avg_win_rate += wl["win"] / len(rule_idx)

        print("Average win rate: {}".format(avg_win_rate))

        return results, reply_dicts


def create_working_dir(args, working_dir):

    if os.path.exists(working_dir):
        print("Attempting to create an existing folder.. hence skipping")
    else:
        os.makedirs(working_dir)
        src = args.rule_dir
        dest = working_dir
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

    print("Created working rule directory at: {}".format(working_dir))
