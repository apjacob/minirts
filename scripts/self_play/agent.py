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
from executor_wrapper import ExecutorWrapper, MultiExecutorWrapper
from executor import Executor
from common_utils import to_device, ResultStat, Logger
from best_models import best_executors, best_coaches
from tqdm import tqdm
from game_utils import *
import copy
import wandb

reward_tuple = [("win", 1), ("loss", -1)]


class Agent:
    def __init__(
        self,
        coach,
        executor,
        device,
        args,
        writer,
        trainable=False,
        exec_sample=False,
        pg="ppo",
        tag="",
    ):

        self.__coach = coach
        self.__executor = executor
        self.__trainable = trainable
        self.device = device
        self.args = args
        self.tb_log = args.tb_log
        self.save_folder = None
        self.tag = tag
        self.best_test_win_pct = 0
        self.tb_writer = writer
        self.traj_dict = defaultdict(list)
        self.result_dict = {}
        self.result = ResultStat("reward", None)
        self.model = self.load_model(self.__coach, self.__executor, self.args)
        self.exec_sample = exec_sample

        print("Using pg {} algorithm".format(args.pg))
        if self.__trainable:
            # if args.split_train:
            #     self.executor_optimizer = optim.Adam(
            #         self.model.executor.parameters(), lr=args.lr
            #     )
            #     self.coach_optimizer = optim.Adam(
            #         self.model.coach.parameters(), lr=args.lr
            #     )
            # else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.sa_buffer = StateActionBuffer(
                max_buffer_size=args.max_table_size, buffer_add_prob=args.sampling_freq
            )
            # wandb.watch(self.model)
            self.pg = pg
        else:
            self.optimizer = None
            self.sa_buffer = None
            self.pg = None

    @property
    def executor(self):
        return self.__executor

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @property
    def coach(self):
        return self.__coach

    def clone(self, model_type="current"):
        if model_type == "current":
            print("Cloning current model.")
            cpy = Agent(
                self.__coach,
                self.__executor,
                self.device,
                self.args,
                self.tb_writer,
                trainable=False,
            )
            cpy.model = self.model
        else:
            raise NameError

        return cpy

    def load_model(self, coach_path, executor_path, args):
        coach_rule_emb_size = getattr(args, "coach_rule_emb_size", 0)
        executor_rule_emb_size = getattr(args, "executor_rule_emb_size", 0)
        inst_dict_path = getattr(args, "inst_dict_path", None)
        coach_random_init = getattr(args, "coach_random_init", False)

        if isinstance(coach_path, str):
            if "onehot" in coach_path:
                coach = ConvOneHotCoach.load(coach_path).to(self.device)
            elif "gen" in coach_path:
                coach = RnnGenerator.load(coach_path).to(self.device)
            else:
                coach = ConvRnnCoach.rl_load(
                    coach_path,
                    coach_rule_emb_size,
                    inst_dict_path,
                    coach_random_init=coach_random_init,
                ).to(self.device)
        else:
            print("Sharing coaches.")
            coach = coach_path
        coach.max_raw_chars = args.max_raw_chars

        if isinstance(executor_path, str):
            executor = Executor.rl_load(
                executor_path, executor_rule_emb_size, inst_dict_path
            ).to(self.device)
        else:
            print("Sharing executors.")
            executor = executor_path

        executor_wrapper = ExecutorWrapper(
            coach,
            executor,
            coach.num_instructions,
            args.max_raw_chars,
            args.cheat,
            args.inst_mode,
        )
        executor_wrapper.train(False)
        return executor_wrapper

    def simulate(self, index, batch):
        with torch.no_grad():
            reply, coach_reply = self.model.forward(batch, self.exec_sample)

            if self.__trainable and self.model.training:
                assert self.sa_buffer is not None

                ## Add coach reply to state-reply dict
                # batch.update(coach_reply['samples'])
                rv = self.sa_buffer.push(index, batch)

            return reply

    def reset(self):
        if self.__trainable and self.model.training:
            assert len(self.sa_buffer) == 0
        else:
            self.sa_buffer = StateActionBuffer(
                max_buffer_size=self.args.max_table_size,
                buffer_add_prob=self.args.sampling_freq,
            )

        self.result = ResultStat("reward", None)
        self.traj_dict = defaultdict(list)
        self.result_dict = {}

    def update_logs(self, index, batch, reply):
        self.result.feed(batch)

        inst = reply["inst"]  ## Get get_inst
        game_ids = batch["game_id"].cpu().numpy()
        terminals = batch["terminal"].cpu().numpy().flatten()
        rewards = batch["reward"].cpu().numpy().flatten()

        # print(len(state_table))
        games_terminated = 0
        for i, g_id in enumerate(game_ids):

            act_dict = self.traj_dict
            act_win_dict = self.result_dict
            dict_key = str(index) + "_" + str(g_id[0])

            act_dict[dict_key].append(self.model.coach.inst_dict.get_inst(inst[i]))

            if terminals[i] == 1:
                games_terminated += 1

                # print("Game {} has terminated.".format(g_id[0]))
                if rewards[i] == 1:
                    act_win_dict[dict_key] = 1
                elif rewards[i] == -1:
                    act_win_dict[dict_key] = -1

        return games_terminated

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def eval_model(self, gen_id, other_agent, num_games=100):
        print("Evaluating model....")
        e_result1, e_result2 = run_eval(
            self.args, self.model, other_agent.model, self.device, num_games
        )
        test_win_pct = e_result1.win / e_result1.num_games

        print(e_result1.log(0))
        print(e_result2.log(0))

        if self.tb_log:
            a1_win = e_result1.win / e_result1.num_games
            a1_loss = e_result1.loss / e_result1.num_games

            if self.args.opponent == "rb":
                a2_win = e_result1.loss / e_result1.num_games
                a2_loss = e_result1.win / e_result1.num_games
            else:
                a2_win = e_result2.win / e_result2.num_games
                a2_loss = e_result2.loss / e_result2.num_games

            wandb.log(
                {
                    "Test/Agent-1/Win": a1_win,
                    "Test/Agent-1/Loss": a1_loss,
                    "Test/Eval_model/Win": a2_win,
                    "Test/Eval_model/Loss": a2_loss,
                },
                step=gen_id,
            )

            self.tb_writer.add_scalar("Test/Agent-1/Win", a1_win, gen_id)
            self.tb_writer.add_scalar("Test/Agent-1/Loss", a1_loss, gen_id)

            self.tb_writer.add_scalar("Test/Eval_model/Win", a2_win, gen_id)
            self.tb_writer.add_scalar("Test/Eval_model/Loss", a2_loss, gen_id)

            if test_win_pct > self.best_test_win_pct:
                self.best_test_win_pct = test_win_pct
                self.save_coach(gen_id)
                self.save_executor(gen_id)
                # wandb.save('{}/*.pt'.format(self.save_folder))
                # wandb.save('{}/*.params'.format(self.save_folder))

        return e_result1, e_result2

    def init_save_folder(self, log_name):

        self.save_folder = os.path.join(self.args.save_folder, log_name)

        if os.path.exists(self.save_folder):
            print("Attempting to create an existing folder.. hence skipping...")
        else:
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

        action_samples = [
            "current_cmd_type",
            "current_cmd_unit_type",
            "current_cmd_x",
            "current_cmd_y",
            "current_cmd_gather_idx",
            "current_cmd_attack_idx",
        ]

        action_samples_dict = {key: "t_" + key for key in action_samples}

        for (g_id, elements) in batches.items():

            for asp in action_samples_dict:
                elements[action_samples_dict[asp]] = elements[asp][1:]

            for (key, element) in elements.items():

                ## Alignment
                if key in action_samples_dict.values():
                    continue
                else:
                    elements[key] = element[:-1]

    def train_executor(self, gen_id, agg_win_batches=None, agg_loss_batches=None):
        assert self.__trainable
        assert self.args.sampling_freq >= 1.0

        if len(self.sa_buffer) or (
            agg_win_batches is not None and agg_loss_batches is not None
        ):
            # self.align_executor_actions(win_batches)
            # self.align_executor_actions(loss_batches)

            if agg_loss_batches is not None and agg_win_batches is not None:
                win_batches = agg_win_batches
                loss_batches = agg_loss_batches
            else:
                win_batches, loss_batches = self.sa_buffer.pop(self.result_dict)

            if self.pg == "vanilla":
                l1_loss, mse_loss, value = self.__update_executor_vanilla(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Executor/RL-Loss": l1_loss,
                            "Loss/Executor/MSE-Loss": mse_loss,
                            "Loss/Executor/Value": value,
                        },
                        step=gen_id,
                    )
                    self.tb_writer.add_scalar("Loss/Executor/RL-Loss", l1_loss, gen_id)
                    self.tb_writer.add_scalar(
                        "Loss/Executor/MSE-Loss", mse_loss, gen_id
                    )
                    self.tb_writer.add_scalar("Loss/Executor/Value", value, gen_id)

            elif self.pg == "ppo":
                action_loss, value_loss, entropy = self.__update_executor_ppo(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Executor/Action-Loss": action_loss,
                            "Loss/Executor/Value-Loss": value_loss,
                            "Loss/Executor/Entropy": entropy,
                        },
                        step=gen_id,
                    )

                    self.tb_writer.add_scalar(
                        "Loss/Executor/Action-Loss", action_loss, gen_id
                    )
                    self.tb_writer.add_scalar(
                        "Loss/Executor/Value-Loss", value_loss, gen_id
                    )
                    self.tb_writer.add_scalar("Loss/Executor/Entropy", entropy, gen_id)
            else:
                raise NotImplementedError
        else:
            print("State-Action Buffer is empty.")

    def __update_executor_vanilla(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss(reduction="none")

        denom = len(win_batches) + len(loss_batches)

        l1_losses = 0
        mse_losses = 0
        total_values = 0

        for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):

            for game_id, batch in batches.items():

                episode_len = batch["reward"].size(0)
                intervals = list(range(0, episode_len, self.args.train_batch_size))
                interval_tuples_iter = zip(intervals, intervals[1:] + [episode_len])

                for (s, e) in interval_tuples_iter:

                    sliced_batch = slice_batch(batch, s, e)

                    (
                        log_prob,
                        all_losses,
                        value,
                    ) = self.model.get_executor_vanilla_rl_train_loss(sliced_batch)
                    l1 = (
                        r * log_prob
                    )  # (r * torch.ones_like(value) - value.detach()) * log_prob
                    l2 = mse(value, r * torch.ones_like(value))

                    l1_mean = -1.0 * l1.sum() / denom
                    mse_loss = 1.0 * l2.sum() / denom
                    policy_loss = l1_mean  # + mse_loss

                    policy_loss.backward()

                    l1_losses += l1_mean.item()
                    mse_losses += mse_loss.item()
                    total_values += value.sum().item() / denom

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model.train()
        return l1_losses, mse_losses, total_values

    def __update_executor_ppo(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss()

        action_loss_mean = 0
        value_loss_mean = 0
        entropy_mean = 0
        num_iters = 0

        ##TODO: Add add discount factor

        for epoch in range(self.args.ppo_epochs):

            for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):

                for game_id, batch in batches.items():
                    episode_len = batch["reward"].size(0)
                    intervals = list(range(0, episode_len, self.args.train_batch_size))
                    interval_tuples_iter = zip(intervals, intervals[1:] + [episode_len])

                    for (s, e) in interval_tuples_iter:
                        sliced_batch = slice_batch(batch, s, e)
                        (
                            log_prob,
                            old_exec_log_probs,
                            entropy,
                            value,
                        ) = self.model.get_executor_ppo_train_loss(sliced_batch)
                        adv = r * torch.ones_like(value) - value.detach()

                        ratio = torch.exp(log_prob - old_exec_log_probs)
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(
                                ratio, 1.0 - self.args.ppo_eps, 1.0 + self.args.ppo_eps
                            )
                            * adv
                        )

                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = 1.0 * mse(value, r * torch.ones_like(value))
                        entropy_loss = -1.0 * entropy  # .mean()

                        self.optimizer.zero_grad()
                        policy_loss = (action_loss + value_loss).mean()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 0.5 * self.args.grad_clip
                        )
                        self.optimizer.step()

                        action_loss_mean += action_loss.item()
                        value_loss_mean += value_loss.item()
                        entropy_mean += entropy_loss  # .item()
                        num_iters += 1

        action_loss_mean = action_loss_mean / num_iters
        value_loss_mean = value_loss_mean / num_iters
        entropy_mean = entropy_mean / num_iters

        self.model.train()
        return action_loss_mean, value_loss_mean, entropy_mean

    def save_executor(self, index):
        assert self.save_folder is not None

        model_file = os.path.join(
            self.save_folder, f"best_exec_checkpoint{self.tag}_{index}.pt"
        )
        print("Saving model exec to: ", model_file)
        self.model.executor.save(model_file)
        wandb.save(model_file)
        wandb.save(model_file + ".params")

        model_file = os.path.join(
            self.save_folder, f"best_exec_checkpoint{self.tag}.pt"
        )
        print("Saving model exec to: ", model_file)
        self.model.executor.save(model_file)
        wandb.save(model_file)
        wandb.save(model_file + ".params")

    ####################
    ## Coach specific ##
    ####################

    def train_coach(self, gen_id):
        assert self.__trainable

        if len(self.sa_buffer):
            win_batches, loss_batches = self.sa_buffer.pop(self.result_dict)

            if self.pg == "vanilla":
                l1_loss, mse_loss, value = self.__update_coach_vanilla(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Coach/Action-Loss": l1_loss,
                            "Loss/Coach/Value-Loss": mse_loss,
                            "Loss/Coach/Value": value,
                        },
                        step=gen_id,
                    )

                    self.tb_writer.add_scalar("Loss/Coach/Action-Loss", l1_loss, gen_id)
                    self.tb_writer.add_scalar("Loss/Coach/Value-Loss", mse_loss, gen_id)
                    self.tb_writer.add_scalar("Loss/Coach/Value", value, gen_id)

            elif self.pg == "ppo":
                action_loss, value_loss, entropy = self.__update_coach_ppo(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Coach/Action-Loss": action_loss,
                            "Loss/Coach/Value-Loss": value_loss,
                            "Loss/Coach/Entropy": entropy,
                        },
                        step=gen_id,
                    )

                    self.tb_writer.add_scalar(
                        "Loss/Coach/Action-Loss", action_loss, gen_id
                    )
                    self.tb_writer.add_scalar(
                        "Loss/Coach/Value-Loss", value_loss, gen_id
                    )
                    self.tb_writer.add_scalar("Loss/Coach/Entropy", entropy, gen_id)
                    return win_batches, loss_batches
            else:
                raise NotImplementedError

        else:
            print("State-Action Buffer is empty.")

    def __update_coach_vanilla(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss(reduction="none")

        denom = np.sum([y["inst"].size()[0] for x, y in win_batches.items()]) + np.sum(
            [y["inst"].size()[0] for x, y in loss_batches.items()]
        )

        l1_loss_mean = 0
        mse_loss_mean = 0

        total_value = 0
        for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):
            # if kind == "win":
            #     denom = np.sum([y['inst'].size()[0] for x, y in win_batches.items()])
            # else:
            #     denom = np.sum([y['inst'].size()[0] for x, y in loss_batches.items()])

            for game_id, batch in batches.items():
                log_prob, value = self.model.get_coach_vanilla_rl_train_loss(batch)
                l1 = (r * torch.ones_like(value) - value.detach()) * log_prob
                l2 = mse(value, r * torch.ones_like(value))

                l1_mean = -1.0 * (l1.sum()) / denom
                mse_loss = 0.1 * (l2.sum()) / denom
                policy_loss = l1_mean + mse_loss

                policy_loss.backward()
                l1_loss_mean += l1_mean.item()
                mse_loss_mean += mse_loss.item()

                total_value += value.sum().item() / denom

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip * 10
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.train()
        return l1_loss_mean, mse_loss_mean, total_value

    def __update_coach_ppo(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss()

        action_loss_mean = 0
        value_loss_mean = 0
        entropy_mean = 0
        num_iters = 0

        ##TODO: Add add discount factor

        for epoch in range(self.args.ppo_epochs):

            for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):

                for game_id, batch in batches.items():
                    episode_len = batch["reward"].size(0)
                    intervals = list(range(0, episode_len, self.args.train_batch_size))
                    interval_tuples_iter = zip(intervals, intervals[1:] + [episode_len])

                    for (s, e) in interval_tuples_iter:
                        sliced_batch = slice_batch(batch, s, e)
                        (
                            log_prob,
                            old_coach_log_probs,
                            entropy,
                            value,
                        ) = self.model.get_coach_ppo_rl_train_loss(sliced_batch)
                        adv = r * torch.ones_like(value) - value.detach()

                        ratio = torch.exp(log_prob - old_coach_log_probs)
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(
                                ratio, 1.0 - self.args.ppo_eps, 1.0 + self.args.ppo_eps
                            )
                            * adv
                        )

                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = 1.0 * mse(value, r * torch.ones_like(value))
                        entropy_loss = 0.0  # -0.001*entropy.mean()

                        self.optimizer.zero_grad()
                        policy_loss = (
                            action_loss + value_loss
                        ).mean()  # + entropy_loss).mean()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 0.5 * self.args.grad_clip
                        )
                        self.optimizer.step()

                        action_loss_mean += action_loss.item()
                        value_loss_mean += value_loss.item()
                        entropy_mean += 0  # entropy_loss.item()
                        num_iters += 1

        action_loss_mean = action_loss_mean / num_iters
        value_loss_mean = value_loss_mean / num_iters
        entropy_mean = entropy_mean / num_iters

        self.model.train()
        return action_loss_mean, value_loss_mean, entropy_mean

    def save_coach(self, index):
        assert self.save_folder is not None

        model_file = os.path.join(
            self.save_folder, f"best_coach_checkpoint{self.tag}_{index}.pt"
        )
        print("Saving model coach to: ", model_file)
        self.model.coach.save(model_file)
        wandb.save(model_file)
        wandb.save(model_file + ".params")

        model_file = os.path.join(
            self.save_folder, f"best_coach_checkpoint{self.tag}.pt"
        )
        print("Saving model coach to: ", model_file)
        self.model.coach.save(model_file)
        wandb.save(model_file)
        wandb.save(model_file + ".params")

    ## Dual RL Training
    def train_both(self, gen_id):
        assert self.__trainable
        assert self.args.sampling_freq >= 1.0

        if len(self.sa_buffer):

            win_batches, loss_batches = self.sa_buffer.pop(self.result_dict)
            # self.align_executor_actions(win_batches)
            # self.align_executor_actions(loss_batches)

            if self.pg == "vanilla":
                l1_loss, mse_loss, value = self.__update_both_vanilla(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Both/RL-Loss": l1_loss,
                            "Loss/Both/MSE-Loss": mse_loss,
                            "Loss/Both/Value": value,
                        },
                        step=gen_id,
                    )
                    self.tb_writer.add_scalar("Loss/Both/RL-Loss", l1_loss, gen_id)
                    self.tb_writer.add_scalar("Loss/Both/MSE-Loss", mse_loss, gen_id)
                    self.tb_writer.add_scalar("Loss/Both/Value", value, gen_id)

            elif self.pg == "ppo":
                action_loss, value_loss, entropy = self.__update_both_ppo(
                    win_batches, loss_batches
                )

                if self.tb_log:
                    wandb.log(
                        {
                            "Loss/Both/Action-Loss": action_loss,
                            "Loss/Both/Value-Loss": value_loss,
                            "Loss/Both/Entropy": entropy,
                        },
                        step=gen_id,
                    )

                    self.tb_writer.add_scalar(
                        "Loss/Both/Action-Loss", action_loss, gen_id
                    )
                    self.tb_writer.add_scalar(
                        "Loss/Both/Value-Loss", value_loss, gen_id
                    )
                    self.tb_writer.add_scalar("Loss/Both/Entropy", entropy, gen_id)
            else:
                raise NotImplementedError
        else:
            print("State-Action Buffer is empty.")

    def __update_both_vanilla(self, win_batches, loss_batches):
        raise NotImplementedError
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss(reduction="none")

        denom = np.sum([y["inst"].size()[0] for x, y in win_batches.items()]) + np.sum(
            [y["inst"].size()[0] for x, y in loss_batches.items()]
        )

        l1_loss_mean = 0
        mse_loss_mean = 0

        total_value = 0
        for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):
            # if kind == "win":
            #     denom = np.sum([y['inst'].size()[0] for x, y in win_batches.items()])
            # else:
            #     denom = np.sum([y['inst'].size()[0] for x, y in loss_batches.items()])

            for game_id, batch in batches.items():
                log_prob, value = self.model.get_coach_vanilla_rl_train_loss(batch)
                l1 = (r * torch.ones_like(value) - value.detach()) * log_prob
                l2 = mse(value, r * torch.ones_like(value))

                l1_mean = -1.0 * (l1.sum()) / denom
                mse_loss = 0.1 * (l2.sum()) / denom
                policy_loss = l1_mean + mse_loss

                policy_loss.backward()
                l1_loss_mean += l1_mean.item()
                mse_loss_mean += mse_loss.item()

                total_value += value.sum().item() / denom

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip * 10
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.train()
        return l1_loss_mean, mse_loss_mean, total_value

    def __update_both_ppo(self, win_batches, loss_batches):
        assert self.__trainable

        self.model.train()
        self.optimizer.zero_grad()

        mse = torch.nn.MSELoss()

        action_loss_mean = 0
        value_loss_mean = 0
        entropy_mean = 0
        num_iters = 0

        ##TODO: Add add discount factor

        for epoch in range(self.args.ppo_epochs):

            for (kind, r), batches in zip(reward_tuple, [win_batches, loss_batches]):

                for game_id, batch in batches.items():
                    episode_len = batch["reward"].size(0)
                    intervals = list(range(0, episode_len, self.args.train_batch_size))
                    interval_tuples_iter = zip(intervals, intervals[1:] + [episode_len])

                    for (s, e) in interval_tuples_iter:
                        sliced_batch = slice_batch(batch, s, e)
                        (
                            c_log_prob,
                            old_coach_log_probs,
                            c_entropy,
                            value,
                        ) = self.model.get_coach_ppo_rl_train_loss(sliced_batch)
                        (
                            e_log_prob,
                            old_exec_log_probs,
                            e_entropy,
                            value,
                        ) = self.model.get_executor_ppo_train_loss(sliced_batch)

                        log_prob = c_log_prob + e_log_prob
                        old_log_prob = old_coach_log_probs + old_exec_log_probs
                        adv = r * torch.ones_like(value) - value.detach()
                        ratio = torch.exp(log_prob - old_log_prob)
                        surr1 = ratio * adv
                        surr2 = (
                            torch.clamp(
                                ratio, 1.0 - self.args.ppo_eps, 1.0 + self.args.ppo_eps
                            )
                            * adv
                        )
                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = 1.0 * mse(value, r * torch.ones_like(value))
                        entropy_loss = 0.0  # -0.001*entropy.mean()

                        self.optimizer.zero_grad()
                        policy_loss = (
                            action_loss + value_loss
                        ).mean()  # + entropy_loss).mean()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 0.5 * self.args.grad_clip
                        )
                        self.optimizer.step()

                        action_loss_mean += action_loss.item()
                        value_loss_mean += value_loss.item()
                        entropy_mean += 0  # entropy_loss.item()
                        num_iters += 1

        action_loss_mean = action_loss_mean / num_iters
        value_loss_mean = value_loss_mean / num_iters
        entropy_mean = entropy_mean / num_iters

        self.model.train()
        return action_loss_mean, value_loss_mean, entropy_mean


class MultiExecutorAgent(Agent):
    def __init__(
        self,
        coach,
        executors,
        device,
        args,
        writer,
        trainable=False,
        exec_sample=False,
        pg="ppo",
        tag="",
    ):
        super().__init__(
            coach, executors, device, args, writer, trainable, exec_sample, pg, tag
        )

    def simulate(self, index, batch):
        with torch.no_grad():
            reply, coach_reply, replies = self.model.forward(batch, self.exec_sample)

            return reply, replies

    def load_model(self, coach_path, executor_paths, args):
        coach_rule_emb_size = getattr(args, "coach_rule_emb_size", 0)
        executor_rule_emb_size = getattr(args, "executor_rule_emb_size", 0)
        inst_dict_path = getattr(args, "inst_dict_path", None)
        coach_random_init = getattr(args, "coach_random_init", False)

        assert isinstance(executor_paths, dict)

        if isinstance(coach_path, str):
            if "onehot" in coach_path:
                coach = ConvOneHotCoach.load(coach_path).to(self.device)
            elif "gen" in coach_path:
                coach = RnnGenerator.load(coach_path).to(self.device)
            else:
                coach = ConvRnnCoach.rl_load(
                    coach_path,
                    coach_rule_emb_size,
                    inst_dict_path,
                    coach_random_init=coach_random_init,
                ).to(self.device)
        else:
            print("Sharing coaches.")
            coach = coach_path
        coach.max_raw_chars = args.max_raw_chars

        executors = {}
        for k, executor_path in executor_paths.items():
            executor = Executor.rl_load(
                executor_path, executor_rule_emb_size, inst_dict_path
            ).to(self.device)
            executors[k] = executor

        executor_wrapper = MultiExecutorWrapper(
            coach,
            executors,
            coach.num_instructions,
            args.max_raw_chars,
            args.cheat,
            args.inst_mode,
        )
        executor_wrapper.train(False)
        return executor_wrapper


def run_eval(args, model1, model2, device, num_games=100):

    num_eval_games = num_games

    result1 = ResultStat("reward", None)
    result2 = ResultStat("reward", None)

    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_options(
        args, [model1.coach.num_instructions, model2.coach.num_instructions]
    )

    if args.opponent == "sp":
        context, act1_dc, act2_dc = init_mt_games(
            num_eval_games, 0, args, ai1_option, ai2_option, game_option
        )
        pbar = tqdm(total=num_eval_games * 2)
    else:
        context, act1_dc, act2_dc = init_mt_games(
            0, num_eval_games, args, ai1_option, ai2_option, game_option
        )
        pbar = tqdm(total=num_eval_games)
    # context, act1_dc, act2_dc = init_games(
    #     num_eval_games, ai1_option, ai2_option, game_option)
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
            if key == "act1":
                batch["actor"] = "act1"
                ## Add batches to state table using sampling before adding
                ## Add based on the game_id

                result1.feed(batch)
                with torch.no_grad():
                    reply, _ = model1.forward(batch)  # , exec_sample=True)

            elif key == "act2":
                batch["actor"] = "act2"
                result2.feed(batch)

                with torch.no_grad():
                    reply, _ = model2.forward(batch)

            else:
                assert False

            dc.set_reply(key, reply)

            game_ids = batch["game_id"].cpu().numpy()
            terminals = batch["terminal"].cpu().numpy().flatten()

            for i, g_id in enumerate(game_ids):
                if terminals[i] == 1:
                    pbar.update(1)

    model1.eval()
    model2.eval()
    pbar.close()

    return result1, result2


def slice_batch(batch, s, e):
    sliced_batch = {}

    for k, v in batch.items():
        if k == "actor":
            sliced_batch[k] = v
        else:
            sliced_batch[k] = v[s:e]

    return sliced_batch
