from utils import (parse_map_from_file, parse_locs, show_args,
                   move)
from naive_agent import SafeAgent, RandomAgent
from search_agent import AStarAgent, DijkstraAgent
from mdp_agent import MDPAgent, HistoryMDPAgent
from pomdp_agent import POMDPAgent, QMDPAgent
from ts_agent import UniformTreeSearchAgent, AsymmetricTreeSearch
from ma_env import MAPF

import argparse
from queue import Queue
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INT_MAX = np.iinfo(np.int64).max


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


def vis_path_layout(dist, fig, ax):
    hmap = ax.matshow(dist)
    fig.colorbar(hmap, ax=ax, location='bottom',
                 ticks=range(0, np.max(dist) + 1, int(np.ceil(np.max(dist) / 10))),
                 shrink=0.7)


def exp_opponents(ax, me, init, layout, dist, opponents, num_sim=1e2):
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
    print(mean, quant05, quant95)
    ax.plot(range(1, dist_max), mean, label='SafeAgent')
    ax.fill_between(range(1, dist_max),
                    np.where(mean - std >= 0, mean - std, 0),
                    mean + std,
                    alpha=0.2, label='std')
    ax.fill_between(range(1, dist_max),
                    quant05,
                    quant95,
                    alpha=0.2, label='quantile')
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

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    args.starts = parse_locs(args.starts)
    args.num_sim = int(args.num_sim)

    return args


if __name__ == '__main__':
    args = get_args()
    show_args(args)

    opponents = [SafeAgent, RandomAgent, AStarAgent]
    me = SafeAgent

    row, col = args.map.shape
    ratio = row // col * args.size
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(ratio * 2, args.size), dpi=100)

    dist = BFS(args.starts['p1'], args.map)
    vis_path_layout(dist, fig, ax[0])
    exp_opponents(ax[1],
                  me, args.starts['p1'], args.map, dist, opponents,
                  num_sim=args.num_sim)
    plt.show()
