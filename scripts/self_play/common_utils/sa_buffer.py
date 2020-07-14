import random
import torch
from collections import defaultdict

class StateActionBuffer:

    def __init__(self, max_buffer_size, buffer_add_prob):
        self.__max_buffer_size = max_buffer_size
        self.__buffer_add_prob = buffer_add_prob
        self.__batch_buffer_dict = defaultdict(dict)

    def __len__(self):
        return len(self.__batch_buffer_dict)

    def _sample(self):
        rnd = random.uniform(0.0, 1.0)
        return rnd <= self.__buffer_add_prob

    def pop(self, wl_dict):

        win_batch_dict = defaultdict(dict)
        loss_batch_dict = defaultdict(dict)

        for (g_id, r) in wl_dict.items():

            if g_id in self.__batch_buffer_dict:
                if r == 1:
                    win_batch_dict[g_id] = self.__batch_buffer_dict.pop(g_id)
                else:
                    loss_batch_dict[g_id] = self.__batch_buffer_dict.pop(g_id)
            else:
                print("Game {} missing from batch buffer".format(g_id))

        return win_batch_dict, loss_batch_dict

    def get(self, wl_dict):

        win_batch_dict = defaultdict(dict)
        loss_batch_dict = defaultdict(dict)

        for (g_id, r) in wl_dict.items():

            if g_id in self.__batch_buffer_dict:
                if r == 1:
                    win_batch_dict[g_id] = self.__batch_buffer_dict[g_id]
                else:
                    loss_batch_dict[g_id] = self.__batch_buffer_dict[g_id]
            else:
                print("Game {} missing from batch buffer".format(g_id))

        return win_batch_dict, loss_batch_dict

    def push(self, gen_id, sr_dict):
        """
        :param gen_id: Generation of the current batch
        :param sr_dict: State-action dictionary
        :return:
        """
        if self._sample():
            self.process_entry(gen_id, sr_dict)
            return 1
        else:
            return -1

    def process_entry(self, gen_id, sr_dict):
        """
        :param gen_id:
        :param sr_dict: Processes a batch dict that is updated with the coach/exec reply
        :return:
        """

        game_ids = sr_dict["game_id"].cpu().numpy()

        for i, g_id_n in enumerate(game_ids):
            ##Need to convert g_ids from tensor to int
            g_id = g_id_n[0]
            full_g_id = str(gen_id) + "_" + str(g_id)

            buffer = self.__batch_buffer_dict
            merge_batch(buffer[full_g_id], i, sr_dict)

def merge_batch(batch_dict_item, index, batch):

    for k, v in batch.items():
        if k == "actor":
            continue

        if k in batch_dict_item:
            batch_dict_item[k] = torch.cat((batch_dict_item[k], v[index].unsqueeze(0)), 0)
        else:
            batch_dict_item[k] = v[index].unsqueeze(0)
