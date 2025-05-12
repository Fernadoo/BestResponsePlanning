from stable_baselines3 import PPO
# from networks import ObsExtractor
# policy_kwargs = dict(
#     features_extractor_class=ObsExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=512,
#         N=5, loc_hidden_dim=1024, belief_hidden_dim=1024,
#         feat_loc_dim=256, feat_belief_dim=256, postproc_hidden_dim=512,
#     ),
#     net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024]),
# )
# meta_policy = PPO.load(
#     'pretrained/MetaPPO_square5a.zip',
#     policy_kwargs=policy_kwargs,
#     custom_objects=dict(policy_kwargs=policy_kwargs)
# )
meta_small = PPO.load('pretrained/MetaPPO_small_8e6.zip')
meta_square = PPO.load('pretrained/MetaPPO_square_3e6.zip')

from utils import (parse_map_from_file, parse_locs, show_args,
                   move)
from cbs_agent import CBSAgent, write_layout_file, write_agent_file, call_solver
from naive_agent import SafeAgent, EnhancedSafeAgent, RandomAgent, ChasingAgent
from search_agent import AStarAgent, DijkstraAgent, astar
from mdp_agent import MDPAgent, HistoryMDPAgent
from pomdp_agent import POMDPAgent, QMDPAgent
from uts_agent import UniformTreeSearchAgent
from mcts_agent import AsymmetricTreeSearch
from meta_agent import MetaAgent
from ma_env import MAPF

import argparse
import numpy as np
import os
import pickle
import random
import re
from functools import partial
from queue import Queue
from collections import namedtuple


from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm

INT_MAX = np.iinfo(np.int64).max
COLORS = list(mcolors.TABLEAU_COLORS)


"""
Exp settings:

MAP | NUM_SIM | SOFT_UPDATE | DEPTH | SAMPLE_EVAL | SAMPLE_BACKUP | MAX_IT | SAMPLE_SELECT

soft_update = (1 - alpha) / (alpha * n - 1),
    where alpha is the desired convergence (0.98), n is the number of empty cells.
"""
SETTINGS = {
    'small': dict(
        num_sim=5e2, soft_update=7e-4,
        depth=2, sample_eval=10, sample_backup=0,  # sample_eval = 0
        max_it=30, sample_select=50,
        meta='pretrained/MetaPPO_small_8e6.zip'),
    'square': dict(
        num_sim=1e3, soft_update=2e-4,
        depth=2, sample_eval=10, sample_backup=0,
        max_it=50, sample_select=50,
        meta='pretrained/MetaPPO_square_3e6.zip'),
    'square4a': dict(
        num_sim=1.5e3, soft_update=2e-4,
        depth=1, sample_eval=5, sample_backup=10,
        max_it=60, sample_select=50),
    'medium20a': dict(
        num_sim=2e3, soft_update=8e-5,
        depth=None, sample_eval=5, sample_backup=None,
        max_it=80, sample_select=80),
    'random50a': dict(
        num_sim=5e3, soft_update=2e-5,
        depth=None, sample_eval=2, sample_backup=None,
        max_it=100, sample_select=125),
}


"""
Renaming agents
"""


def AStar_Agent(label, goal, config):
    return AStarAgent(label, goal)


def Safe_Agent(label, goal, config):
    return SafeAgent(label, goal)


def EnhancedSafe_Agent(label, goal, config):
    return EnhancedSafeAgent(label, goal)


def MDPAgentFixedBelief(label, goal, config):
    return MDPAgent(label, goal, belief_update=False)


def MDPAgentUpdateBelief(label, goal, config):
    return MDPAgent(label, goal, belief_update=True, soft_update=config['soft_update'])


def MetaAgentFixedBelief(label, goal, config):
    return MetaAgent(label, goal,
                     belief_update=False,
                     meta_policy=PPO.load(config['meta']))


def MetaAgentUpdateBelief(label, goal, config):
    return MetaAgent(label, goal,
                     belief_update=True, soft_update=config['soft_update'],
                     meta_policy=PPO.load(config['meta']))


def UniformTSAgentMeta(label, goal, config):
    nn_rewards = {
        'collision': 10,
        'goal': 10
    }
    return UniformTreeSearchAgent(label, goal,
                                  belief_update=True, soft_update=config['soft_update'],
                                  depth=config['depth'],
                                  node_eval='NN', nn_estimator=PPO.load(config['meta']),
                                  reward_scheme=nn_rewards)


def CBSAgentFixedBelief(label, goal, config):
    return CBSAgent(label, goal,
                    belief_update=False,
                    sample_eval=config['sample_eval'])


def CBSAgentUpdateBelief(label, goal, config):
    return CBSAgent(label, goal,
                    belief_update=True, soft_update=config['soft_update'],
                    sample_eval=config['sample_eval'])


def UniformTSAgentCBS(label, goal, config):
    return UniformTreeSearchAgent(label, goal,
                                  belief_update=True, soft_update=config['soft_update'],
                                  depth=config['depth'],
                                  node_eval='CBS', sample_eval=config['sample_eval'],
                                  sample_backup=config['sample_backup'])


def MCTSAgentCBSucb(label, goal, config):
    return AsymmetricTreeSearch(label, goal,
                                belief_update=True, soft_update=config['soft_update'],
                                max_it=config['max_it'],
                                node_eval='CBS', sample_eval=config['sample_eval'],
                                sample_select=config['sample_select'],
                                pUCB=False)


def MCTSAgentCBSpuct(label, goal, config):
    return AsymmetricTreeSearch(label, goal,
                                belief_update=True, soft_update=config['soft_update'],
                                max_it=config['max_it'],
                                node_eval='CBS', sample_eval=config['sample_eval'],
                                sample_select=config['sample_select'],
                                pUCB=True)


# def UniformTSAgentD2(label, goal, config):
#     return UniformTreeSearchAgent(label, goal, belief_update=True, soft_update=1e-6,
#                                   depth=2, node_eval='HEU-C',
#                                   check_repeated_states=True)


# def AsymmetricTreeSearchE3(label, goal, config):
#     return AsymmetricTreeSearch(label, goal, belief_update=True, soft_update=1e-6,
#                                 max_it=1e3, node_eval='HEU-C')


def RandomAgent08(label, goal, config):
    return RandomAgent(label, goal, p=0.8)


def RandomAgent06(label, goal, config):
    return RandomAgent(label, goal, p=0.8)


def RandomAgent04(label, goal, config):
    return RandomAgent(label, goal, p=0.8)


def ChasingAgentp10(label, goal, config):
    return ChasingAgent(label, goal, p_chase=1)


def ChasingAgentp08(label, goal, config):
    return ChasingAgent(label, goal, p_chase=0.8)


def ChasingAgentp06(label, goal, config):
    return ChasingAgent(label, goal, p_chase=0.6)


def ChasingAgentp04(label, goal, config):
    return ChasingAgent(label, goal, p_chase=0.4)


"""
Experiment scripts
"""


def generate_tests(num_agents, possible_oppo_types, layout, num_sim=1e3, seed=618):
    random.seed(seed)
    num_sim = int(num_sim)
    empty_cells = np.array(np.where(layout == 0)).T

    random_tests = []
    for sim in tqdm(range(num_sim)):
        rand_starts = random.sample(empty_cells.tolist(), k=num_agents)
        rand_starts = [tuple(s) for s in rand_starts]
        rand_goals = random.sample(empty_cells.tolist(), k=num_agents)
        rand_goals = [tuple(s) for s in rand_goals]
        rand_oppo = random.choices(possible_oppo_types, k=num_agents - 1)
        random_tests.append((rand_starts, rand_goals, rand_oppo))

    return random_tests


def run_experiments(me, num_agents, layout, random_tests, config, num_sim=1e3, seed=618):
    random.seed(seed)
    num_sim = int(num_sim)

    step_record = []
    collision_record = []
    stuck_record = []
    for sim in tqdm(range(num_sim)):
        rand_starts, rand_goals, rand_oppo = random_tests[sim]
        # print(rand_starts, rand_goals, rand_oppo)
        players = []
        for i in range(num_agents):
            if i == 0:
                players.append(me(label=0, goal=rand_goals[0], config=config))
            else:
                players.append(rand_oppo[i - 1](label=i, goal=rand_goals[i], config=config))
        game = MAPF(players, rand_starts, rand_goals, layout, stop_on=0)
        _, steps, collisions, stuck = game.run()
        # print(f"steps: {steps[0]}, collisions: {collisions[0]}")
        step_record.append(steps[0])
        collision_record.append(collisions[0])
        stuck_record.append(stuck)

    return dict(step_record=np.array(step_record),
                collision_record=np.array(collision_record),
                stuck_record=np.array(stuck_record))


def run_selfplay(me, num_agents, layout, random_tests, config, num_sim=1e3, seed=618):
    random.seed(seed)
    num_sim = int(num_sim)

    step_record = []
    collision_record = []
    stuck_record = []
    for sim in tqdm(range(num_sim)):
        rand_starts, rand_goals, _ = random_tests[sim]
        players = []
        for i in range(num_agents):
            players.append(me(label=i, goal=rand_goals[i], config=config))
        game = MAPF(players, rand_starts, rand_goals, layout, stop_on=None)
        _, steps, collisions, stuck = game.run()
        step_record.append(np.mean(steps))
        collision_record.append(np.mean(collisions))
        stuck_record.append(stuck)

    return dict(step_record=np.array(step_record),
                collision_record=np.array(collision_record),
                stuck_record=np.array(stuck_record))


def func2multiproc_func(q, func, *args):
    ret = func(*args)
    q.put((ret['step_record'], ret['collision_record'], ret['stuck_record']))


def run_multi_proc(func,
                   me, num_agents, layout, random_tests, config, num_sim=1e3, seed=618,
                   num_proc=8):
    import multiprocessing as mp

    # divide the random tests
    subtasks = []
    unit_len = len(random_tests) // num_proc
    for i in range(num_proc):
        if i < num_proc - 1:
            subtasks.append(random_tests[i * unit_len: (i + 1) * unit_len])
        else:
            subtasks.append(random_tests[i * unit_len:])

    assert sum(list(map(len, subtasks))) == len(random_tests)

    processes = []
    q = mp.Queue()
    for i in range(num_proc):
        subproc = mp.Process(target=func2multiproc_func,
                             args=(q, func,
                                   me, num_agents, layout, subtasks[i], config, len(subtasks[i]), seed))
        subproc.start()
        processes.append(subproc)

    for subproc in processes:
        subproc.join()
        print(subproc)

    merged_step_records = []
    merged_collision_records = []
    merged_stuck_records = []
    while not q.empty():
        step_record, collision_record, stuck_record = q.get()
        merged_step_records.append(step_record)
        merged_collision_records.append(collision_record)
        merged_stuck_records.append(stuck_record)
    merged_step_records = np.concatenate(merged_step_records)
    merged_collision_records = np.concatenate(merged_collision_records)
    merged_stuck_records = np.concatenate(merged_stuck_records)
    print(np.mean(merged_step_records),
          np.mean(merged_collision_records),
          np.mean(merged_stuck_records))

    return dict(step_record=merged_step_records,
                collision_record=merged_collision_records,
                stuck_record=merged_stuck_records)


def vis_map(layout, fig, ax):
    cmap = mcolors.ListedColormap(['white', 'grey'])
    ax.pcolor(layout, cmap=cmap, edgecolors='k', linewidths=0.1)
    ax.invert_yaxis()


def plot_performance(data, fig, ax, COEF):
    fig.subplots_adjust(right=0.75)
    step_list = []
    collision_list = []
    agg_mean_list = []
    agg_std_list = []
    ticks = []
    for i, name in enumerate(data):
        print(name, len(data[name]['step_record']))
        # print(np.where(data[name]['step_record'] == np.max(data[name]['step_record'])))
        # return np.where(data[name]['step_record'] == np.max(data[name]['step_record']))[0].tolist()
        penalized_steps = data[name]['step_record'] * (1 - data[name]['stuck_record'])\
            + 4 * COEF * data[name]['stuck_record']
        # print(data[name]['step_record'])
        # print(data[name]['stuck_record'])
        # print(penalized_steps)
        step_list.append(np.mean(penalized_steps))
        collision_list.append(np.mean(data[name]['collision_record'] > 0))
        # agg = []
        # for j in range(len(data[name]['step_record'])):
        #     num_steps = data[name]['step_record'][j]
        #     num_collisions = data[name]['collision_record'][j]
        #     if num_collisions > 0:
        #         agg.append(COEF)
        #     else:
        #         agg.append(num_steps)
        agg = penalized_steps * (data[name]['collision_record'] == 0)\
            + 4 * COEF * (data[name]['collision_record'] > 0)
        agg_mean_list.append(np.mean(agg))
        agg_std_list.append(np.std(agg))
        # print(agg.tolist())
        ticks.append(name)
    # exit()

    labels = [f"{mean.round(2)} ({std.round(2)})" for mean, std in zip(agg_mean_list, agg_std_list)]
    color = 'steelblue'
    ax.set_ylabel('Aggregated', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    bars = ax.bar(ticks, agg_mean_list, width=0.5, yerr=None, color=color)
    ax.set_xticklabels(ticks, rotation=20, fontsize=8, ha='right')
    ax.bar_label(bars, labels, padding=3, fontsize=6)

    ax_step = ax.twinx()
    color = 'orange'
    ax_step.set_ylabel('Steps', color=color)
    ax_step.tick_params(axis='y', labelcolor=color)
    ax_step.plot(ticks, step_list, marker='o', color=color)
    y0, y1 = ax_step.get_yticks()[:2]
    for x, y in enumerate(step_list):
        ax_step.text(x, y + (y1 - y0) / 5, s=y.round(2),
                     ha='center', va='center', fontsize=6, color=color)

    ax_coll = ax.twinx()
    color = 'limegreen'
    ax_coll.spines.right.set_position(("axes", 1.1))
    ax_coll.set_ylabel('Collisions', color=color)
    ax_coll.tick_params(axis='y', labelcolor=color)
    ax_coll.plot(ticks, collision_list, marker='x', color=color)
    y0, y1 = ax_coll.get_yticks()[:2]
    for x, y in enumerate(collision_list):
        ax_coll.text(x, y + (y1 - y0) / 5, s=y.round(2),
                     ha='center', va='center', fontsize=6, color=color)


def compute_astar_and_plot(layout, random_tests, fig, ax):
    ref_record = []
    for test in tqdm(random_tests):
        rand_starts, rand_goals, _ = test
        plan = astar(rand_starts[0], rand_goals[0], layout)
        ref_record.append(len(plan))
    mean = np.mean(ref_record)
    std = np.std(ref_record)
    ax.hlines(mean, *ax.get_xlim(), linestyle=':', color='r')
    ax.text(ax.get_xticks()[0], mean - 0.1, f'{mean.round(2)} ({std.round(2)})',
            color='r', verticalalignment='center', horizontalalignment='right', fontsize=8)
    # ax.fill_between(np.arange(*ax.get_xlim(), 0.1), mean + std, mean - std, alpha=0.3)


def compute_cbs_and_plot(layout, random_tests, fig, ax, load_from=None):
    ref_record = []
    if load_from:
        ref_record = load_from
    else:
        prefix = 'cbs-solve/tmp'
        hash_id = 'ref'
        num_agents = None
        layout_file = write_layout_file(layout, hash_id, layout_file_prefix=prefix)
        for test in tqdm(random_tests):
            rand_starts, rand_goals, _ = test
            num_agents = len(rand_starts)
            agent_file = write_agent_file(rand_starts, rand_goals, hash_id, agent_file_prefix=prefix)
            sol_file = call_solver(layout_file, agent_file, len(rand_starts), hash_id,
                                   sol_file_prefix=prefix,
                                   timeout=120,
                                   subopt=1 if layout.shape[0] < 25 else 1.2)
            # parse solution
            soc = 0
            N = 0
            with open(sol_file, 'r') as sf:
                line = sf.readline()
                while line:  # e.g., Agent 0: (16,5)->(17,5)->(17,6)->
                    chunks = re.split(': |->', line)
                    steps = chunks[1: -1]
                    soc += (len(steps) - 1)
                    N += 1
                    line = sf.readline()
            assert N == num_agents
            ref_record.append(soc / N)

    mean = np.mean(ref_record)
    std = np.std(ref_record)
    ax.hlines(mean, *ax.get_xlim(), linestyle=':', color='red')
    ax.text(ax.get_xticks()[0], mean - 0.1, f'{mean.round(2)} ({std.round(2)})',
            color='red', verticalalignment='center', horizontalalignment='right', fontsize=8)
    # ax.fill_between(np.arange(*ax.get_xlim(), 0.1), mean + std, mean - std, alpha=0.3)

    return ref_record


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation.'
    )

    parser.add_argument('--map', dest='map', type=str,
                        help='Specify a map')
    parser.add_argument('--agents', dest='agents', type=int, default=0,
                        help='Specify the number of agents')
    parser.add_argument('--cfg', dest='cfg', type=str,
                        help='Specify a parameter configuration')
    parser.add_argument('--tests', dest='tests', type=int, nargs='+',
                        help='Specify the type of the controlled agent')
    parser.add_argument('--oppo-type', dest='oppo_type', type=str, default='rational',
                        help='Specify the type of opponents among [malicous, rational, evolving]')
    parser.add_argument('--num-sim', dest='num_sim', type=float, default=1e3,
                        help='Specify the number of tesing simulations')
    parser.add_argument('--size', dest='size', type=int, default=7,
                        help='Specify the figsize')
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Plot the experimental results')
    parser.add_argument('--load', dest='load', action='store_true',
                        help='Load the existing experimental results')
    parser.add_argument('--prefix', dest='prefix', type=str, default='./',
                        help='Specify the prefix of exp_file loading')
    parser.add_argument('--mp', dest='mp', type=int, default=0,
                        help='Specify the number of multiprocessing')

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    args.num_sim = int(args.num_sim)

    return args


if __name__ == '__main__':
    args = get_args()
    show_args(args)

    COEF = min(args.map.shape) * 1  # *1 for small/square, *2 for medium/large
    print(f"Penalty coef for collisions: {COEF}, max {COEF * 4}")
    # exit()

    CONFIG = SETTINGS[args.cfg]
    print(CONFIG)

    opponents = ['']
    if args.oppo_type == 'malicious':
        opponents = [ChasingAgentp10, ChasingAgentp08, ChasingAgentp06, ChasingAgentp04]
    elif args.oppo_type == 'rational':
        opponents = [AStar_Agent, RandomAgent08, RandomAgent06, Safe_Agent]
    print(opponents)

    candidate_agents = {
        0: Safe_Agent,
        1: AStar_Agent,

        2: MDPAgentFixedBelief,
        3: MDPAgentUpdateBelief,

        4: MetaAgentFixedBelief,
        5: MetaAgentUpdateBelief,
        6: UniformTSAgentMeta,

        7: CBSAgentFixedBelief,
        8: CBSAgentUpdateBelief,
        9: UniformTSAgentCBS,
        10: MCTSAgentCBSucb,
        11: MCTSAgentCBSpuct,

        12: EnhancedSafe_Agent
    }
    me_agents = []
    for i in args.tests:
        me_agents.append(candidate_agents[i])
    print(me_agents)

    random_tests = generate_tests(num_agents=args.agents,
                                  possible_oppo_types=opponents,
                                  layout=args.map,
                                  num_sim=args.num_sim)

    # print(random_tests)
    if not args.load:
        for me in me_agents:
            print(f'--- TESTING {me.__name__} ---')
            if args.oppo_type == 'self':
                if args.mp:
                    info = run_multi_proc(
                        run_selfplay,
                        me,
                        num_agents=args.agents,
                        layout=args.map,
                        random_tests=random_tests,
                        config=CONFIG,
                        num_sim=args.num_sim,
                        num_proc=args.mp)
                else:
                    info = run_selfplay(me, num_agents=args.agents,
                                        layout=args.map,
                                        random_tests=random_tests,
                                        config=CONFIG,
                                        num_sim=args.num_sim)
            else:
                if args.mp:
                    info = run_multi_proc(
                        run_experiments,
                        me,
                        num_agents=args.agents,
                        layout=args.map,
                        random_tests=random_tests,
                        config=CONFIG,
                        num_sim=args.num_sim,
                        num_proc=args.mp)
                else:
                    info = run_experiments(me, num_agents=args.agents,
                                           layout=args.map,
                                           random_tests=random_tests,
                                           config=CONFIG,
                                           num_sim=args.num_sim)
            with open(f'{args.prefix}_{args.oppo_type}/INFO_{me.__name__}.pkl', 'wb') as pklf:
                pickle.dump(info, pklf)

    if args.plot:
        data = dict()
        for me in me_agents:
            with open(f'{args.prefix}_{args.oppo_type}/INFO_{me.__name__}.pkl', 'rb') as pklf:
                data[f'{me.__name__}'] = pickle.load(pklf)

        row, col = args.map.shape
        aspect = row / col * args.size
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(aspect, args.size), dpi=100)
        vis_map(args.map, fig1, ax1)
        fig1.savefig(f'{args.prefix}_{args.oppo_type}/map.png', dpi=200)

        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(args.size * 1.5, args.size), dpi=100)
        oot = plot_performance(data, fig2, ax2, COEF)

        # for t in oot[:5]:
        #     print(random_tests[t])
        # exit()

        reference = None
        if args.oppo_type == 'self':
            if os.path.exists(f'{args.prefix}_{args.oppo_type}/CBS.pkl'):
                with open(f'{args.prefix}_{args.oppo_type}/CBS.pkl', 'rb') as pklf:
                    load_from = pickle.load(pklf)
                compute_cbs_and_plot(args.map, random_tests, fig2, ax2, load_from=load_from)
            else:
                record = compute_cbs_and_plot(args.map, random_tests, fig2, ax2, load_from=None)
                with open(f'{args.prefix}_{args.oppo_type}/CBS.pkl', 'wb') as pklf:
                    pickle.dump(record, pklf)
        else:
            compute_astar_and_plot(args.map, random_tests, fig2, ax2)

        fig2.savefig(f'{args.prefix}_{args.oppo_type}/mean_std_rand.pdf', dpi=150)
        plt.show()
