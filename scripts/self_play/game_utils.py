import os

from set_path import append_sys_path
append_sys_path()

import random
import tube
import minirts


def init_games(num_games, ai1_option, ai2_option, game_option, *, act_name='act'):
    # print('ai1 option:')
    # print(ai1_option.info())
    # print('ai2 option:')
    # print(ai2_option.info())
    # print('game option:')
    # print(game_option.info())

    batchsize = min(32, max(num_games // 2, 1))
    act1_dc = tube.DataChannel(act_name+'1', batchsize, 1)
    act2_dc = tube.DataChannel(act_name+'2', batchsize, 1)
    context = tube.Context()
    idx2utype = [
        minirts.UnitType.SPEARMAN,
        minirts.UnitType.SWORDMAN,
        minirts.UnitType.CAVALRY,
        minirts.UnitType.DRAGON,
        minirts.UnitType.ARCHER,
    ]

    if game_option.seed == 777:
        print("Using random seeds...")
        seed = random.randint(1, 123456)
    else:
        seed = game_option.seed

    for i in range(num_games):
        g_option = minirts.RTSGameOption(game_option)
        g_option.seed = seed + i
        g_option.game_id = str(i)
        if game_option.save_replay_prefix:
            g_option.save_replay_prefix = game_option.save_replay_prefix + "_0_" + str(i)

        g = minirts.RTSGame(g_option)
        bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act1_dc)
        bot2 = minirts.CheatExecutorAI(ai2_option, 0, None, act2_dc)
        # utype = idx2utype[i % len(idx2utype)]
        # bot2 = minirts.MediumAI(ai2_option, 0, None, utype, False)
        # p1dict[i] = []
        # p2dict[i] = []
        g.add_bot(bot1)
        g.add_bot(bot2)
        context.push_env_thread(g)

    return context, act1_dc, act2_dc

def get_game_option(args):
    game_option = minirts.RTSGameOption()
    game_option.seed = args.seed
    game_option.max_tick = args.max_tick
    game_option.no_terrain = args.no_terrain
    game_option.resource = args.resource
    game_option.resource_dist = args.resource_dist
    game_option.fair = args.fair
    game_option.save_replay_freq = args.save_replay_freq
    game_option.save_replay_per_games = args.save_replay_per_games
    game_option.lua_files = args.lua_files
    game_option.num_games_per_thread = args.game_per_thread
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units

    save_dir = os.path.abspath(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    game_option.save_replay_prefix = save_dir + '/'

    return game_option


def get_ai_options(args, num_instructions):
    options = []
    for i in range(2):
        ai_option = minirts.AIOption()
        ai_option.t_len = 1
        ai_option.fs = args.frame_skip
        ai_option.fow = args.fow
        ai_option.use_moving_avg = args.use_moving_avg
        ai_option.moving_avg_decay = args.moving_avg_decay
        ai_option.num_resource_bins = args.num_resource_bins
        ai_option.resource_bin_size = args.resource_bin_size
        ai_option.max_num_units = args.max_num_units
        ai_option.num_prev_cmds = args.num_prev_cmds
        ai_option.num_instructions = num_instructions[i]
        ai_option.max_raw_chars = args.max_raw_chars
        ai_option.verbose = args.verbose
        options.append(ai_option)

    return options[0], options[1]