# Copyright (c) Facebook, Inc. and its affiliates.# All rights reserved.## This source code is licensed under the license found in the# LICENSE file in the root directory of this source tree.#import argparseimport osimport sysimport pprintfrom set_path import append_sys_pathappend_sys_path()import torchimport randomimport tubefrom agent import Agentfrom pytube import DataChannelManagerfrom common_utils import StateActionBufferfrom torch.utils.tensorboard import SummaryWriterimport minirtsimport numpy as npimport picklefrom collections import defaultdictimport torch.optim as optimfrom rnn_coach import ConvRnnCoachfrom onehot_coach import ConvOneHotCoachfrom rnn_generator import RnnGeneratorfrom itertools import groupbyfrom executor_wrapper import ExecutorWrapperfrom executor import Executorfrom common_utils import to_device, ResultStat, Loggerfrom best_models import best_executors, best_coachesfrom tqdm import tqdmfrom game import *import wandbdef parse_args():    parser = argparse.ArgumentParser(description='human coach')    parser.add_argument('--seed', type=int, default=1)    parser.add_argument('--num_thread', type=int, default=1)    parser.add_argument('--num_iter', type=int, default=10)    parser.add_argument('--train_batch_size', type=int, default=32)    parser.add_argument('--tag', type=str, default='')    parser.add_argument('--sampling_freq', type=float, default=0.4)    parser.add_argument('--sp_factor', type=int, default=2)    parser.add_argument('--eval_factor', type=int, default=10)    parser.add_argument('--max_table_size', type=int, default=100)    parser.add_argument('--game_per_thread', type=int, default=1)    parser.add_argument('--gpu', type=int, default=0)    root = os.path.dirname(        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    default_lua = os.path.join(root, 'game/game_MC/lua')    parser.add_argument('--lua_files', type=str, default=default_lua)    # ai1 option    parser.add_argument('--frame_skip', type=int, default=50)    parser.add_argument('--fow', type=int, default=1)    parser.add_argument('--use_moving_avg', type=int, default=1)    parser.add_argument('--moving_avg_decay', type=float, default=0.98)    parser.add_argument('--num_resource_bins', type=int, default=11)    parser.add_argument('--resource_bin_size', type=int, default=50)    parser.add_argument('--max_num_units', type=int, default=50)    parser.add_argument('--num_prev_cmds', type=int, default=25)    # TOOD: add max instruction span    parser.add_argument('--max_raw_chars', type=int, default=200)    parser.add_argument('--verbose', action='store_true')    parser.add_argument('--inst_mode', type=str, default='full') # can be full/good/better    # game option    parser.add_argument('--max_tick', type=int, default=int(2e4))    parser.add_argument('--no_terrain', action='store_true')    parser.add_argument('--resource', type=int, default=500)    parser.add_argument('--resource_dist', type=int, default=4)    parser.add_argument('--fair', type=int, default=0)    parser.add_argument('--save_replay_freq', type=int, default=0)    parser.add_argument('--save_replay_per_games', type=int, default=1)    parser.add_argument('--save_dir', type=str, default='matches2/dev')    parser.add_argument('--save_folder', type=str, default='/home/ubuntu/minirts/scripts/self_play/save')    parser.add_argument('--coach_load_file', type=str, default='/home/ubuntu/minirts/scripts/self_play/save/')    parser.add_argument('--coach_reload', type=int, default=0)    parser.add_argument('--exec_load_file', type=str, default='/home/ubuntu/minirts/scripts/self_play/save/')    parser.add_argument('--exec_reload', type=int, default=0)    parser.add_argument('--tb_log', type=int, default=1)    # full vision    parser.add_argument('--cheat', type=int, default=0)    parser.add_argument('--coach1', type=str, default='')    parser.add_argument('--executor1', type=str, default='')    parser.add_argument('--coach2', type=str, default='')    parser.add_argument('--executor2', type=str, default='')    parser.add_argument('--lr', type=float, default=1e-2)    parser.add_argument('--beta1', type=float, default=0.9)    parser.add_argument('--beta2', type=float, default=0.999)    parser.add_argument('--grad_clip', type=float, default=0.5)    parser.add_argument('--pg', type=str, default='vanilla')    parser.add_argument('--ppo_eps', type=float, default=0.2)    parser.add_argument('--ppo_epochs', type=int, default=3)    args = parser.parse_args()    return argsdef self_play():    global device    args = parse_args()    print('args:')    pprint.pprint(vars(args))    os.environ['LUA_PATH'] = os.path.join(args.lua_files, '?.lua')    print('lua path:', os.environ['LUA_PATH'])    if not os.path.exists(args.save_dir):        os.makedirs(args.save_dir)    wandb.init(name="coach_selfplay", project="adapt-minirts", sync_tensorboard=True)    wandb.config.update(args)    if args.coach_reload:        print("Reloading coach model.... ")        args.coach1 = args.coach_load_file        _coach1 = os.path.basename(args.coach1).replace(".pt", "")    else:        _coach1 = args.coach1        args.coach1 = best_coaches[args.coach1]    if args.exec_reload:        print("Reloading executor model.... ")        args.executor1 = args.exec_load_file        _executor1 = os.path.basename(args.executor1).replace(".pt", "")    else:        _executor1 = args.executor1        args.executor1 = best_executors[args.coach1]    log_name = "coach_sp_adaptive_coach_c1_type={}__e1_type={}_lr={}__num_iter={}__num_thread={}__sp_factor={}__pg={}_{}_{}". \        format(_coach1, args.coach2, _executor1, args.executor2, args.lr, args.num_iter, args.num_thread,               args.tag, random.randint(1111, 9999))    writer = SummaryWriter(comment=log_name)    args.coach2     = best_coaches[args.coach2]    args.executor2  = best_executors[args.executor2]    logger_path = os.path.join(args.save_dir, 'train.log')    sys.stdout = Logger(logger_path)    device = torch.device('cuda:%d' % args.gpu)    sp_agent = Agent(coach=args.coach1, executor=args.executor1, device=device, args=args, writer=writer, trainable=True, pg=args.pg)    sp_agent.init_save_folder(log_name)    bc_agent = Agent(coach=args.coach2, executor=args.executor2, device=device, args=args, writer=writer, trainable=False)    print("Progress: ")    ## Create Save folder:    for index in range(args.num_iter):        if index % args.eval_factor == 0:            sp_agent.eval_model(index, bc_agent)        game = Game(sp_agent, bc_agent, index, args)        agent1, agent2  = game.start()        agent1.train()        agent2.train()        pbar = tqdm(total=args.num_thread * 2)        while not game.finished():            data = game.get_input()            if len(data) == 0:                continue            for key in data:                # print(key)                batch = to_device(data[key], device)                if key == 'act1':                    batch['actor'] = 'act1'                    reply = agent1.simulate(index, batch)                    t_count = agent1.update_logs(index, batch, reply)                elif key == 'act2':                    batch['actor'] = 'act2'                    reply = agent2.simulate(index, batch)                    t_count = agent2.update_logs(index, batch, reply)                else:                    assert False                game.set_reply(key, reply)                pbar.update(t_count)        agent1.train_coach(index)        game.print_logs(index)        game.terminate()        del game        pbar.close()    writer.close()if __name__ == '__main__':    self_play()