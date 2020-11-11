import os

from set_path import append_sys_path

append_sys_path()

import random
import tube
import minirts


def init_games(num_games, ai1_option, ai2_option, game_option, *, act_name="act"):
    # print('ai1 option:')
    # print(ai1_option.info())
    # print('ai2 option:')
    # print(ai2_option.info())
    # print('game option:')
    # print(game_option.info())

    batchsize = min(32, max(num_games // 2, 1))
    act1_dc = tube.DataChannel(act_name + "1", batchsize, 1)
    act2_dc = tube.DataChannel(act_name + "2", batchsize, 1)
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

    for i in range(num_games):
        g_option = minirts.RTSGameOption(game_option)
        if game_option.seed == 777:
            print("Using random seeds...")
            seed = random.randint(1, 123456)
        else:
            seed = game_option.seed

        g_option.seed = seed + i
        g_option.game_id = str(i)
        if game_option.save_replay_prefix:
            g_option.save_replay_prefix = (
                game_option.save_replay_prefix + "_0_" + str(i)
            )

        g = minirts.RTSGame(g_option)
        bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act1_dc)
        bot2 = minirts.CheatExecutorAI(ai2_option, 0, None, act2_dc)
        # utype = idx2utype[i % len(idx2utype)]
        # bot2 = minirts.MediumAI(ai2_option, 0, None, utype, False)

        g.add_bot(bot1)
        g.add_bot(bot2)
        context.push_env_thread(g)

    return context, act1_dc, act2_dc


def init_mt_games(
    num_sp,
    num_rb,
    args,
    ai1_option,
    ai2_option,
    game_option,
    *,
    act_name="act",
    viz=False
):
    # print('ai1 option:')
    # print(ai1_option.info())
    # print('ai2 option:')
    # print(ai2_option.info())
    # print('game option:')
    # print(game_option.info())

    if game_option.seed == 777:
        print("Using random seeds...")

    total_games = num_sp + num_rb
    batchsize = min(32, max(total_games // 2, 1))

    act1_dc = tube.DataChannel(act_name + "1", batchsize, 1)
    act2_dc = tube.DataChannel(act_name + "2", batchsize, 1)
    context = tube.Context()
    idx2utype = [
        minirts.UnitType.SPEARMAN,
        minirts.UnitType.SWORDMAN,
        minirts.UnitType.CAVALRY,
        minirts.UnitType.DRAGON,
        minirts.UnitType.ARCHER,
    ]

    game_id = 0
    rnd_num = random.randint(1, num_rb - 1)
    for i in range(num_rb):
        if game_option.seed == 777:
            seed = random.randint(1, 123456)
        else:
            seed = game_option.seed

        bot1, g = create_game(act1_dc, ai1_option, game_option, game_id, seed)

        rule_type = i % len(idx2utype)
        utype = idx2utype[rule_type]
        bot2 = minirts.MediumAI(ai2_option, 0, None, utype, 1)  # Utype + tower

        g.add_bot(bot1)
        g.add_bot(bot2)

        if viz and i == rnd_num:
            g.add_default_spectator()

        context.push_env_thread(g)
        game_id += 1

    for i in range(num_sp):
        if game_option.seed == 777:
            seed = random.randint(1, 123456)
        else:
            seed = game_option.seed

        bot1, g = create_game(act1_dc, ai1_option, game_option, game_id, seed)
        bot2 = minirts.CheatExecutorAI(ai2_option, 0, None, act2_dc)

        g.add_bot(bot1)
        g.add_bot(bot2)
        context.push_env_thread(g)
        game_id += 1

    return context, act1_dc, act2_dc


def init_botvbot(
    bot1idx,
    bot2idx,
    num_games,
    args,
    ai1_option,
    ai2_option,
    game_option,
    *,
    act_name="act",
    viz=False
):
    # print('ai1 option:')
    # print(ai1_option.info())
    # print('ai2 option:')
    # print(ai2_option.info())
    # print('game option:')
    # print(game_option.info())
    total_games = num_games
    batchsize = min(32, max(total_games // 2, 1))

    act1_dc = tube.DataChannel(act_name + "1", batchsize, 1)
    act2_dc = tube.DataChannel(act_name + "2", batchsize, 1)
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

    game_id = 0
    rnd_num = random.randint(1, num_games - 1)
    for i in range(num_games):
        if game_option.seed == 777:
            seed = random.randint(1, 123456)
        else:
            seed = game_option.seed

        g_option = minirts.RTSGameOption(game_option)
        g_option.seed = seed + i
        g_option.game_id = str(i)
        g = minirts.RTSGame(g_option)

        bot1 = minirts.MediumAI(ai1_option, 0, None, idx2utype[bot1idx], 1)
        bot2 = minirts.MediumAI(
            ai2_option, 0, None, idx2utype[bot2idx], 1
        )  # Utype + tower

        g.add_bot(bot1)
        g.add_bot(bot2)

        if viz and i == rnd_num:
            g.add_default_spectator()

        context.push_env_thread(g)
        game_id += 1

    return context, act1_dc, act2_dc


def init_vbot(
    botidx,
    num_games,
    args,
    ai1_option,
    ai2_option,
    game_option,
    *,
    act_name="act",
    viz=False
):
    # print('ai1 option:')
    # print(ai1_option.info())
    # print('ai2 option:')
    # print(ai2_option.info())
    # print('game option:')
    # print(game_option.info())
    total_games = num_games
    batchsize = min(32, max(total_games // 2, 1))

    act1_dc = tube.DataChannel(act_name + "1", batchsize, 1)
    act2_dc = tube.DataChannel(act_name + "2", batchsize, 1)
    context = tube.Context()
    idx2utype = [
        minirts.UnitType.SWORDMAN,
        minirts.UnitType.SPEARMAN,
        minirts.UnitType.CAVALRY,
        minirts.UnitType.ARCHER,
        minirts.UnitType.DRAGON,
    ]

    if game_option.seed == 777:
        print("Using random seeds...")

    game_id = 0
    rnd_num = random.randint(1, num_games - 1)
    for i in range(num_games):
        if game_option.seed == 777:
            seed = random.randint(1, 123456)
        else:
            seed = game_option.seed

        bot1, g = create_game(act1_dc, ai1_option, game_option, game_id, seed)

        utype = idx2utype[botidx]
        bot2 = minirts.MediumAI(ai2_option, 0, None, utype, 1)  # Utype + tower

        g.add_bot(bot1)
        g.add_bot(bot2)

        if viz and i == rnd_num:
            g.add_default_spectator()

        context.push_env_thread(g)
        game_id += 1

    return context, act1_dc, act2_dc


def create_game(act1_dc, ai1_option, game_option, i, seed):
    g_option = minirts.RTSGameOption(game_option)
    g_option.seed = seed + i
    g_option.game_id = str(i)
    if game_option.save_replay_prefix:
        g_option.save_replay_prefix = game_option.save_replay_prefix + "_0_" + str(i)
    g = minirts.RTSGame(g_option)
    bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act1_dc)
    return bot1, g


def get_game_option(args, lua_files=None):

    game_option = minirts.RTSGameOption()
    game_option.seed = args.seed
    game_option.max_tick = args.max_tick
    game_option.no_terrain = args.no_terrain
    game_option.resource = args.resource
    game_option.resource_dist = args.resource_dist
    game_option.fair = args.fair
    game_option.save_replay_freq = args.save_replay_freq
    game_option.save_replay_per_games = args.save_replay_per_games

    if lua_files is None:
        game_option.lua_files = args.lua_files
    else:
        game_option.lua_files = lua_files

    game_option.num_games_per_thread = args.game_per_thread
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units

    save_dir = os.path.abspath(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    game_option.save_replay_prefix = save_dir + "/"

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
