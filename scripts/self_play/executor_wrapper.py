# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

from common_utils import assert_eq, assert_lt
from utils import convert_to_raw_instruction
from instruction_encoder import is_word_based


def format_reply(batch, coach_reply, executor_reply):
    reply = coach_reply.copy()
    reply.update(executor_reply)
    reply['num_unit'] = batch['num_army']
    return reply


class ExecutorWrapper(nn.Module):
    def __init__(self, coach, executor, num_insts, max_raw_chars, cheat, inst_mode):
        super().__init__()
        self.coach = coach
        self.executor = executor
        assert self.executor.inst_dict._idx2inst == self.coach.inst_dict._idx2inst

        self.num_insts = num_insts
        self.max_raw_chars = max_raw_chars
        self.cheat = cheat
        self.inst_mode = inst_mode
        self.prev_inst = ''

    def _get_human_instruction(self, batch):
        assert_eq(batch['prev_inst'].size(0), 1)
        device = batch['prev_inst'].device

        inst = input('Please input your instruction\n')
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
            'inst': inst_idx.unsqueeze(1),
            'inst_pi': torch.ones(1, self.num_insts).to(device) / self.num_insts,
            'cont': inst_cont.unsqueeze(1),
            'cont_pi': torch.ones(1, 2).to(device) / 2,
            'raw_inst': raw_inst
        }

        return inst, inst_len, inst_cont, reply

    def forward(self, batch):
        if self.coach is not None:
            #assert not self.coach.training
            coach_input = self.coach.format_coach_input(batch)
            word_based = is_word_based(self.executor.args.inst_encoder_type)

            # if batch['actor'] == 'act2':
            #     inst_mode = 'mask'
            #     # print("Actor 2 is using custom mask...")
            # else:
            #     inst_mode = self.inst_mode

            inst_mode = self.inst_mode

            inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
                coach_input, inst_mode, word_based)
        else:
            inst, inst_len, inst_cont, coach_reply = self._get_human_instruction(batch)

        # if batch['actor']=='act1':
        #     for i in inst.cpu().numpy()[0]:
        #         if i==927:
        #             break
        #         else:
        #             print(self.coach.inst_dict._idx2word[i], end=" ")
        #     print("")


        #assert not self.executor.training
        executor_input = self.executor.format_executor_input(
            batch, inst, inst_len, inst_cont)
        executor_reply = self.executor.compute_prob(executor_input)

        reply = format_reply(batch, coach_reply, executor_reply)

        return reply, log_prob_reply

    def get_coach_rl_train_loss(self, batch):

        assert self.coach.training

        coach_input = self.coach.format_coach_input(batch)
        word_based = is_word_based(self.executor.args.inst_encoder_type)
        inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
            coach_input, self.inst_mode, word_based)

        ## Replacing with training samples
        log_prob_reply['samples'] = {'inst': batch['inst'], 'cont': batch['cont']}
        value = log_prob_reply['value'] #coach_reply['value']
        loss = self.coach.sampler.get_log_prob(log_prob_reply['probs'], log_prob_reply['samples'])

        return loss, value


    #### Executor train forward ####

    def exec_train_forward(self, batch):
        if self.coach is not None:
            #assert not self.coach.training
            coach_input = self.coach.format_coach_input(batch)
            word_based = is_word_based(self.executor.args.inst_encoder_type)

            # if batch['actor'] == 'act2':
            #     inst_mode = 'mask'
            #     # print("Actor 2 is using custom mask...")
            # else:
            #     inst_mode = self.inst_mode

            inst_mode = self.inst_mode

            inst, inst_len, inst_cont, coach_reply, log_prob_reply = self.coach.sample(
                coach_input, inst_mode, word_based)
        else:
            inst, inst_len, inst_cont, coach_reply = self._get_human_instruction(batch)

        # if batch['actor']=='act1':
        #     for i in inst.cpu().numpy()[0]:
        #         if i==927:
        #             break
        #         else:
        #             print(self.coach.inst_dict._idx2word[i], end=" ")
        #     print("")


        #assert not self.executor.training
        executor_input = self.executor.format_executor_input(
            batch, inst, inst_len, inst_cont)
        executor_reply = self.executor.compute_prob(executor_input)

        reply = format_reply(batch, coach_reply, executor_reply)

        return reply, log_prob_reply
