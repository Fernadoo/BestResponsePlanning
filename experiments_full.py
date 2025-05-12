from stable_baselines3 import PPO
meta_policy = PPO.load('pretrained/MetaPPO_small_8e6.zip')

from utils import (parse_map_from_file, parse_locs, show_args,
                   move)
from naive_agent import SafeAgent, RandomAgent
from search_agent import AStarAgent, DijkstraAgent
from mdp_agent import MDPAgent, HistoryMDPAgent
from pomdp_agent import POMDPAgent, QMDPAgent
from ts_agent import UniformTreeSearchAgent, AsymmetricTreeSearch
from meta_agent import MetaAgent
from ma_env import MAPF

import argparse
import pickle
from queue import Queue
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm

INT_MAX = np.iinfo(np.int64).max
COLORS = list(mcolors.TABLEAU_COLORS)


###################
# Renaming agents #
###################


def MDPAgentFixedBelief(label, goal):
    return MDPAgent(label, goal, belief_update=False)


def MDPAgentUpdateBelief(label, goal):
    return MDPAgent(label, goal, belief_update=True)


def UniformTreeSearchAgentD2(label, goal):
    return UniformTreeSearchAgent(label, goal, belief_update=True,
                                  depth=2, node_eval='HEU-C',
                                  check_repeated_states=True)


def AsymmetricTreeSearchE3(label, goal):
    return AsymmetricTreeSearch(label, goal, belief_update=True,
                                max_it=1e3, node_eval='HEU-C')


def MetaAgentFixedBelief(label, goal):
    return MetaAgent(label, goal, meta_policy, belief_update=False, verbose=False)


def MetaAgentUpdateBelief(label, goal):
    return MetaAgent(label, goal, meta_policy, belief_update=True, verbose=False)


def UniformTSAgentD2Meta(label, goal):
    nn_rewards = {
        'collision': 10,
        'goal': 10
    }
    return UniformTreeSearchAgent(label, goal,
                                  belief_update=True, depth=2, node_eval='NN',
                                  nn_estimator=meta_policy,
                                  reward_scheme=nn_rewards)


###############
# Experiments #
###############


def BFS(init, layout):
    Node = namedtuple('BFSNode',
                      ['gValue', 'PrevAction', 'Loc'])
    nrows = len(layout)
    ncols = len(layout[0])
    num_empty = nrows * ncols - np.sum(layout)

    def get_successors(node):
        g, prev_action, curr_loc = node
        successors = []
        for a in range(5):
            succ_loc = move(curr_loc, a)
            if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols) or\
                    layout[succ_loc] == 1:
                continue
            succ_node = Node(g + 1, a, succ_loc)
            successors.append(succ_node)
        return successors

    visited = []
    dist = np.zeros(shape=layout.shape, dtype=int)
    dist[np.where(layout == 1)] = -1
    num_visited = 0
    q = Queue()
    q.put(Node(0, 0, init))
    while not q.empty() and num_visited < num_empty:
        curr_node = q.get()
        if curr_node.Loc in visited:
            continue
        num_visited += 1
        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
        visited.append(curr_node.Loc)
        dist[tuple(curr_node.Loc)] = curr_node.gValue
    if num_visited < num_empty:
        # raise RuntimeError("No BFS plan found!")
        print("Certain area unreachable")
    return dist


def exp_opponents(me, init, layout, dist, opponents, num_sim=1e2):
    goals_min_to_max = dict()
    dist_max = np.max(dist)
    for d in range(1, dist_max):
        goals_min_to_max[d] = np.transpose(np.where(dist == d))

    num_sim = int(num_sim)
    path_min_to_max = []
    for g in range(1, dist_max):
        num_sample = (
            goals_min_to_max[g].shape[0] * num_sim
            // max(map(lambda x: len(x), goals_min_to_max.values()))
        )
        rand_self_goals_idx = np.random.choice(goals_min_to_max[g].shape[0], num_sample)
        rand_self_goals = goals_min_to_max[g][rand_self_goals_idx]

        rand_oppo = np.random.choice(opponents, num_sample)

        empty_cell = np.transpose(np.where(layout == 0))
        rand_oppo_inits_idx = np.random.choice(empty_cell.shape[0], num_sample)
        rand_oppo_inits = empty_cell[rand_oppo_inits_idx]
        rand_oppo_goals_idx = np.random.choice(empty_cell.shape[0], num_sample)
        rand_oppo_goals = empty_cell[rand_oppo_goals_idx]

        record = []
        for it in tqdm(range(num_sample)):
            if tuple(init) == tuple(rand_oppo_inits[it]) or\
                    tuple(rand_self_goals[it]) == tuple(rand_oppo_goals[it]):
                continue
            starts = (tuple(init), tuple(rand_oppo_inits[it]))
            goals = (tuple(rand_self_goals[it]), tuple(rand_oppo_goals[it]))
            agents = [me(0, goals[0]), rand_oppo[it](1, goals[1])]
            game = MAPF(agents, starts, goals, args.map)
            _, steps = game.run()
            record.append(steps[0])
        path_min_to_max.append(record)

    mean = np.array(list(map(np.mean, path_min_to_max)))
    std = np.array(list(map(np.std, path_min_to_max)))
    quant05 = np.array(list(map(lambda l: np.quantile(l, 0.05), path_min_to_max)))
    quant95 = np.array(list(map(lambda l: np.quantile(l, 0.95), path_min_to_max)))

    return dict(zip(['path_min_to_max', 'mean', 'std', 'quant05', 'quant95'],
                    [path_min_to_max, mean, std, quant05, quant95]))


def vis_path_layout(dist, fig, ax):
    hmap = ax.matshow(dist)
    fig.colorbar(hmap, ax=ax, location='bottom',
                 ticks=range(0, np.max(dist) + 1, int(np.ceil(np.max(dist) / 10))),
                 shrink=0.7)


def plot_performance(dist, data, fig, ax):
    dist_max = np.max(dist)
    for i, name in enumerate(data):
        path_min_to_max, mean, std, quant05, quant95 = data[name].values()
        ax.plot(range(1, dist_max), mean, label=name, color=COLORS[i])
        # ax.fill_between(range(1, dist_max),
        #                 range(1, dist_max),
        #                 mean + std,
        #                 alpha=0.2, color=COLORS[i])
        # ax.fill_between(range(1, dist_max),
        #                 quant05,
        #                 quant95,
        #                 alpha=0.2, label='quantile')
    ax.plot(range(1, dist_max), range(1, dist_max), linestyle=':')
    ax.legend()


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent Planning.'
    )

    parser.add_argument('--map', dest='map', type=str,
                        help='Specify a map')
    parser.add_argument('--starts', dest='starts', type=str, nargs='+',
                        help='Specify the starts for each agent,'
                             'e.g. 2_0 0_2')
    parser.add_argument('--num-sim', dest='num_sim', type=float, default=1e3,
                        help='Specify the figsize')
    parser.add_argument('--size', dest='size', type=int, default=7,
                        help='Specify the figsize')
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Plot the experimental results')
    parser.add_argument('--load', dest='load', action='store_true',
                        help='Load the existing experimental results')
    parser.add_argument('--prefix', dest='prefix', type=str, default='./',
                        help='Specify the prefix of exp_file loading')

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    args.starts = parse_locs(args.starts)
    args.num_sim = int(args.num_sim)

    return args


if __name__ == '__main__':
    args = get_args()
    show_args(args)

    opponents = [SafeAgent, RandomAgent, AStarAgent]
    me_agents = [
        # SafeAgent,
        # MDPAgentFixedBelief,
        # MDPAgentUpdateBelief,
        # UniformTreeSearchAgentD2,
        # AsymmetricTreeSearchE3,
        MetaAgentFixedBelief,
        MetaAgentUpdateBelief,
        UniformTSAgentD2Meta,
    ]

    dist = BFS(args.starts['p1'], args.map)
    print(dist)
    # exit()

    if not args.load:
        for me in me_agents:
            print(f'--- TESTING {me.__name__} ---')
            info = exp_opponents(me, args.starts['p1'], args.map, dist, opponents,
                                 num_sim=args.num_sim)
            with open(f'{args.prefix}/INFO_{me.__name__}.pkl', 'wb') as pklf:
                pickle.dump(info, pklf)

    if args.plot:
        data = dict()
        for me in me_agents:
            with open(f'{args.prefix}/INFO_{me.__name__}.pkl', 'rb') as pklf:
                data[f'{me.__name__}'] = pickle.load(pklf)

        row, col = args.map.shape
        ratio = row // col * args.size
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(ratio * 2, args.size), dpi=100)
        vis_path_layout(dist, fig, ax[0])
        plot_performance(dist, data, fig, ax[1])
        # plt.savefig(f'{args.prefix}/mean_std_best.png', dpi=200)
        plt.show()
