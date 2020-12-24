# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.categorical import Categorical
import copy
from common_utils import assert_eq, assert_lt
from utils import convert_to_raw_instruction
from instruction_encoder import is_word_based
from common_utils.global_consts import *
from executor import compute_log_prob

CMD_TARGET_IDLE = 0
CMD_TARGET_GATHER = 1
CMD_TARGET_ATTACK = 2
CMD_TARGET_BUILD_BUILDING = 3
CMD_TARGET_BUILD_UNIT = 4
CMD_TARGET_MOVE = 5
CMD_TARGET_CONT = 6
NUM_CMD_TARGET_TYPE = 7


def format_reply(batch, coach_reply, executor_reply):
    reply = coach_reply.copy()
    reply.update(executor_reply)
    reply["num_unit"] = batch["num_army"]
    return reply


def construct_samples(executor_reply):

    onehot_reply = {}
    sample_reply = {}

    for (k, v) in executor_reply.items():
        one_hot = OneHotCategorical(v).sample()
        onehot_reply[k] = one_hot
        sample_reply["sample_" + k] = Categorical(one_hot).sample()

    pre_log_prob_sum, pre_log_probs = compute_log_prob(sample_reply, executor_reply)
    return pre_log_prob_sum, pre_log_probs, onehot_reply, sample_reply


class ExecutorWrapper(nn.Module):
    def __init__(self, coach, executor, num_insts, max_raw_chars, cheat, inst_mode):
        super().__init__()
        self.coach = coach
        self.executor = executor
        if coach:
            assert self.executor.inst_dict._idx2inst == self.coach.inst_dict._idx2inst

        self.num_insts = num_insts
        self.max_raw_chars = max_raw_chars
        self.cheat = cheat
        self.inst_mode = inst_mode
        self.prev_inst = ""

    def _get_human_instruction(self, batch):
        assert_eq(batch["prev_inst"].size(0), 1)
        device = batch["prev_inst"].device

        inst = input("Please input your instruction\n")
        import pdb

        pdb.set_trace()
        # inst = 'build peasant'

        inst_idx = torch.zeros((1,)).long().to(device)
        inst_idx[0] = self.executor.inst_dict.get_inst_idx(inst)
        inst_cont = torch.zeros((1,)).long().to(device)
        if len(inst) == 0:
            # inst = batch['prev_inst']
            inst = self.prev_inst
            inst_cont[0] = 1

        self.prev_inst = inst
        raw_inst = convert_to_raw_instruction(inst, self.max_raw_chars)
        inst, inst_len = self.executor.inst_dict.parse(inst, True)
        inst = torch.LongTensor(inst).unsqueeze(0).to(device)
        inst_len = torch.LongTensor([inst_len]).to(device)
        raw_inst = torch.LongTensor([raw_inst]).to(device)

        reply = {
            "inst": inst_idx.unsqueeze(1),
            "inst_pi": torch.ones(1, self.num_insts).to(device) / self.num_insts,
            "cont": inst_cont.unsqueeze(1),
            "cont_pi": torch.ones(1, 2).to(device) / 2,
            "raw_inst": raw_inst,
        }

        return inst, inst_len, inst_cont, reply

    def forward(self, batch, exec_sample=False):
        if self.coach is not None:
            # assert not self.coach.training
            coach_input = self.coach.format_coach_input(batch)
            word_based = is_word_based(self.executor.args.inst_encoder_type)

            inst_mode = self.inst_mode

            inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
                coach_input, inst_mode, word_based
            )
        else:
            inst, inst_len, inst_cont, coach_reply = self._get_human_instruction(batch)

        # assert not self.executor.training
        executor_input = self.executor.format_executor_input(
            batch, inst, inst_len, inst_cont
        )
        executor_reply = self.executor.compute_prob(executor_input)

        if exec_sample:
            (
                pre_log_prob_sum,
                pre_log_probs,
                one_hot_reply,
                sample_reply,
            ) = construct_samples(executor_reply)

            batch["log_prob_sum"] = pre_log_prob_sum
            batch.update(pre_log_probs)
            batch.update(one_hot_reply)
            batch.update(sample_reply)

            executor_reply = one_hot_reply
            ####

        reply = format_reply(batch, coach_reply, executor_reply)
        batch.update(reply)

        ## Added more elements to batch
        batch["e_inst"] = inst
        batch["e_inst_len"] = inst_len
        batch["e_inst_cont"] = inst_cont

        ## Add coach old log probs for PPO
        old_coach_log_probs = self.coach.sampler.get_log_prob(
            log_prob_reply["probs"], log_prob_reply["samples"]
        )
        batch["old_coach_log_probs"] = old_coach_log_probs.detach()

        return reply, log_prob_reply

    def get_coach_vanilla_rl_train_loss(self, batch):

        assert self.coach.training

        coach_input = self.coach.format_coach_input(batch)
        word_based = is_word_based(self.executor.args.inst_encoder_type)
        inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
            coach_input, self.inst_mode, word_based
        )

        ## Replacing with training samples
        log_prob_reply["samples"] = {"inst": batch["inst"], "cont": batch["cont"]}
        value = log_prob_reply["value"]  # coach_reply['value']
        loss = self.coach.sampler.get_log_prob(
            log_prob_reply["probs"], log_prob_reply["samples"]
        )

        return loss, value

    def get_coach_ppo_rl_train_loss(self, batch):

        assert self.coach.training

        coach_input = self.coach.format_coach_input(batch)
        word_based = is_word_based(self.executor.args.inst_encoder_type)
        inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
            coach_input, self.inst_mode, word_based
        )

        ## Get new policy log_probs
        log_prob_reply["samples"] = {"inst": batch["inst"], "cont": batch["cont"]}
        log_probs = self.coach.sampler.get_log_prob(
            log_prob_reply["probs"], log_prob_reply["samples"]
        )
        value = log_prob_reply["value"]
        old_log_probs = batch["old_coach_log_probs"]
        dist_entropy = Categorical(log_prob_reply["probs"]["inst_pi"]).entropy()

        return log_probs, old_log_probs, dist_entropy, value

    #### Executor train forward ####
    def log_probs(self, batch, exec_reply):
        cmd_type = batch["current_cmd_type"]
        cmd_type_prob = exec_reply["cmd_type_prob"]

        gather_idx = batch["current_cmd_gather_idx"]
        gather_idx_prob = exec_reply["gather_idx_prob"]
        gather_idx_mask = cmd_type == CMD_TARGET_GATHER

        attack_idx = batch["current_cmd_attack_idx"]
        attack_idx_prob = exec_reply["attack_idx_prob"]
        attack_idx_mask = cmd_type == CMD_TARGET_ATTACK

        unit_type = batch["current_cmd_unit_type"]
        unit_type_prob = exec_reply["unit_type_prob"]
        unit_type_mask = cmd_type == CMD_TARGET_BUILD_UNIT

        loc_x = batch["current_cmd_x"]
        loc_y = batch["current_cmd_y"]

        loc = loc_y * 32 + loc_x

        building_type = batch["current_cmd_unit_type"]
        building_type_prob = exec_reply["building_type_prob"]
        building_mask = cmd_type == CMD_TARGET_BUILD_BUILDING
        building_loc_prob = exec_reply["building_loc_prob"]

        move_loc_prob = exec_reply["move_loc_prob"]
        move_loc_mask = cmd_type == CMD_TARGET_MOVE

        cmd_type_prob_ = cmd_type_prob.gather(2, cmd_type.unsqueeze(2)).squeeze(2)
        cmd_type_log_prob = cmd_type_prob_.log()

        gather_idx_prob_ = gather_idx_prob.gather(2, gather_idx.unsqueeze(2)).squeeze(2)
        gather_idx_log_prob = gather_idx_prob_.log()

        attack_idx_prob_ = attack_idx_prob.gather(2, attack_idx.unsqueeze(2)).squeeze(2)
        attack_idx_log_prob = attack_idx_prob_.log()

        unit_type_prob_ = unit_type_prob.gather(2, unit_type.unsqueeze(2)).squeeze(2)
        unit_type_log_prob = unit_type_prob_.log()

        building_type_prob_ = building_type_prob.gather(
            2, building_type.unsqueeze(2)
        ).squeeze(2)
        building_type_log_prob = building_type_prob_.log()

        building_loc_prob_ = building_loc_prob.gather(2, loc.unsqueeze(2)).squeeze(2)
        building_loc_log_prob = building_loc_prob_.log()

        move_loc_prob_ = move_loc_prob.gather(2, loc.unsqueeze(2)).squeeze(2)
        move_loc_log_prob = move_loc_prob_.log()

        log_probs = {
            "cmd_type": cmd_type_log_prob,
            "gather_idx": gather_idx_log_prob,
            "attack_idx": attack_idx_log_prob,
            "unit_type": unit_type_log_prob,
            "building_type": building_type_log_prob,
            "building_loc": building_loc_log_prob,
            "move_loc": move_loc_log_prob,
        }

        masks = {
            "gather_idx": gather_idx_mask,
            "attack_idx": attack_idx_mask,
            "unit_type": unit_type_mask,
            "building_type": building_mask,
            "building_loc": building_mask,
            "move_loc": move_loc_mask,
        }

        return log_probs, masks

    def get_executor_vanilla_rl_train_loss(self, batch):
        assert self.executor.training and self.coach.training

        coach_input = self.coach.format_coach_input(batch)
        word_based = is_word_based(self.executor.args.inst_encoder_type)
        _, _, _, _, log_prob_reply = self.coach.sample(
            coach_input, self.inst_mode, word_based
        )

        inst = batch["e_inst"]
        inst_len = batch["e_inst_len"]
        inst_cont = batch["e_inst_cont"]

        # assert not self.executor.training
        executor_input = self.executor.format_rl_executor_input(
            batch, inst, inst_len, inst_cont
        )
        executor_reply = self.executor.compute_prob(executor_input)
        log_prob_sum, all_log_probs = compute_log_prob(batch, executor_reply)

        # log_prob, all_losses = self.executor.compute_rl_log_probs(executor_input)
        value = log_prob_reply["value"]
        # log_probs, masks = self.log_probs(batch, executor_reply)

        return log_prob_sum, all_log_probs, value

    def get_executor_ppo_train_loss(self, batch):
        assert self.executor.training and self.coach.training

        coach_input = self.coach.format_coach_input(batch)
        word_based = is_word_based(self.executor.args.inst_encoder_type)
        _, _, _, _, log_prob_reply = self.coach.sample(
            coach_input, self.inst_mode, word_based
        )

        inst = batch["e_inst"]
        inst_len = batch["e_inst_len"]
        inst_cont = batch["e_inst_cont"]

        # assert not self.executor.training
        executor_input = self.executor.format_rl_executor_input(
            batch, inst, inst_len, inst_cont
        )
        executor_reply = self.executor.compute_prob(executor_input)
        log_prob_sum, all_log_probs = compute_log_prob(batch, executor_reply)

        # log_prob, all_losses = self.executor.compute_rl_log_probs(executor_input)
        value = log_prob_reply["value"]
        old_exec_log_probs = batch["log_prob_sum"]
        log_prob = log_prob_sum
        entropy = 0
        # log_probs, masks = self.log_probs(batch, executor_reply)

        return log_prob, old_exec_log_probs, entropy, value


class MultiExecutorWrapper(ExecutorWrapper):
    def __init__(self, coach, executors, num_insts, max_raw_chars, cheat, inst_mode):
        assert isinstance(executors, dict)
        self.executors = executors
        super().__init__(
            coach, executors["bc"], num_insts, max_raw_chars, cheat, inst_mode
        )

    def forward(self, batch, exec_sample=False):
        replies = {}
        if self.coach is not None:
            # assert not self.coach.training
            coach_input = self.coach.format_coach_input(batch)
            word_based = is_word_based(self.executor.args.inst_encoder_type)

            inst_mode = self.inst_mode

            inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
                coach_input, inst_mode, word_based
            )
        else:
            inst, inst_len, inst_cont, coach_reply = self._get_human_instruction(batch)

        replies["bc_coach"] = move_to_cpu(coach_reply)
        # assert not self.executor.training
        executor_input = self.executor.format_executor_input(
            batch, inst, inst_len, inst_cont
        )
        executor_reply = self.executor.compute_prob(executor_input)

        if exec_sample:
            (
                pre_log_prob_sum,
                pre_log_probs,
                one_hot_reply,
                sample_reply,
            ) = construct_samples(executor_reply)

            batch["log_prob_sum"] = pre_log_prob_sum
            batch.update(pre_log_probs)
            batch.update(one_hot_reply)
            batch.update(sample_reply)
            replies["bc_executor"] = {
                "one_hot_reply": move_to_cpu(one_hot_reply),
                "executor_reply": move_to_cpu(executor_reply),
            }
            executor_reply = one_hot_reply
            ####

        main_reply = format_reply(batch, coach_reply, executor_reply)
        batch.update(main_reply)

        ## Added more elements to batch
        batch["e_inst"] = inst
        batch["e_inst_len"] = inst_len
        batch["e_inst_cont"] = inst_cont

        ## Add coach old log probs for PPO
        old_coach_log_probs = self.coach.sampler.get_log_prob(
            log_prob_reply["probs"], log_prob_reply["samples"]
        )
        batch["old_coach_log_probs"] = old_coach_log_probs.detach()

        for key, executor in self.executors.items():
            if key == "bc":
                continue

            executor_input = executor.format_executor_input(
                batch, inst, inst_len, inst_cont
            )
            executor_reply = executor.compute_prob(executor_input)

            if exec_sample:
                (
                    pre_log_prob_sum,
                    pre_log_probs,
                    one_hot_reply,
                    sample_reply,
                ) = construct_samples(executor_reply)
                ####

            replies[key] = {
                "one_hot_reply": move_to_cpu(one_hot_reply),
                "executor_reply": move_to_cpu(executor_reply),
            }

        return main_reply, log_prob_reply, replies


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_cpu(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_cpu(v))
        return res
    else:
        raise TypeError("Invalid type for move_to")
