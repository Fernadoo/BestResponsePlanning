from stable_baselines3 import PPO
meta_small = PPO.load('pretrained/MetaPPO_small_8e6.zip')
meta_square = PPO.load('pretrained/MetaPPO_square_3e6.zip')

from naive_agent import SafeAgent, EnhancedSafeAgent, RandomAgent, ChasingAgent
from search_agent import AStarAgent, DijkstraAgent
from mdp_agent import MDPAgent, HistoryMDPAgent
from pomdp_agent import POMDPAgent, QMDPAgent
from uts_agent import UniformTreeSearchAgent
from mcts_agent import AsymmetricTreeSearch
from cbs_agent import CBSAgent
from meta_agent import MetaAgent
from ma_env import MAPF
from animator import Animation
from utils import parse_map_from_file, parse_locs, show_args

import argparse
import random

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent Planning.'
    )

    parser.add_argument('--agents', dest='agents', type=int, default=0,
                        help='Specify the number of agents')
    parser.add_argument('--map', dest='map', type=str,
                        help='Specify a map')
    parser.add_argument('--starts', dest='starts', type=str, nargs='+',
                        help='Specify the starts for each agent,'
                             'e.g. 2_0 0_2, or just `random`')
    parser.add_argument('--goals', dest='goals', type=str, nargs='+',
                        help='Specify the goals for each agent,'
                             'e.g. 2_0 0_2, or just `random`')
    parser.add_argument('--vis', dest='vis', action='store_true',
                        help='Visulize the process')
    parser.add_argument('--save', dest='save', type=str,
                        help='Specify the path to save the animation')

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    empty_cells = np.array(np.where(args.map == 0)).T.tolist()
    if 'random' in args.starts:
        candidate = random.sample(empty_cells, k=args.agents)
        args.starts = list(map(tuple, candidate))
    else:
        args.starts = parse_locs(args.starts)
    if 'random' in args.goals:
        candidate = random.sample(empty_cells, k=args.agents)
        args.goals = list(map(tuple, candidate))
    else:
        args.goals = parse_locs(args.goals)
    if args.save:
        args.save = 'results/' + args.save

    return args


def show_hist(history):
    action_list = ['stop', 'up', 'right', 'down', 'left']
    for t, info in enumerate(history):
        actions, locations = info
        if t == 0:
            print(f'T{t}: start from {locations}')
        else:

            print(f'T{t}: '
                  f'actions: {list(map(lambda a: action_list[a], actions))}\t'
                  f'locations: {locations}')


if __name__ == '__main__':

    args = get_args()
    show_args(args)

    agents = []

    """
    Agent 0
    """
    agents.append(AStarAgent(0, args.goals[0]))
    # agents.append(MDPAgent(0, args.goals[0], belief_update=True, verbose=True))
    # agents.append(QMDPAgent(0, args.goals[0]))
    # agents.append(MetaAgent(0, args.goals[0], belief_update=True, verbose=False,
    #                         meta_policy=meta_square))  # `meta_small` or `meta_square`
    # agents.append(CBSAgent(0, args.goals[0], belief_update=False, sample_eval=10, verbose=False))
    # agents.append(UniformTreeSearchAgent(0, args.goals[0],
    #                                      belief_update=True, soft_update=2e-4,
    #                                      depth=2, node_eval='CBS',
    #                                      sample_eval=5,
    #                                      sample_backup=0,
    #                                      verbose=True))
    # agents.append(AsymmetricTreeSearch(0, args.goals[0],
    #                                    belief_update=True, verbose=True, soft_update=8e-5,
    #                                    max_it=100,
    #                                    node_eval='CBS',
    #                                    sample_eval=5,
    #                                    sample_select=125,
    #                                    pUCB=True))
    # agents.append(ChasingAgent(0, args.goals[0], p_chase=0.5))  # need to specify stop_on
    # agents.append(SafeAgent(0, args.goals[0]))
    # agents.append(EnhancedSafeAgent(0, args.goals[0]))
    # agents.append(RandomAgent(0, args.goals[0], p=0.5))
    # agents.append(DijkstraAgent(0, args.goals[args.agents[0]]))
    # nn_rewards = {
    #     'illegal': 1,
    #     'normal': 1,
    #     'collision': 3000,
    #     'goal': 10
    # }
    # agents.append(UniformTreeSearchAgent(0, args.goals[0],
    #                                      belief_update=True, soft_update=2e-4,
    #                                      depth=2, node_eval='NN',
    #                                      verbose=True,
    #                                      nn_estimator=meta_square,  # `meta_small` or `meta_square`
    #                                      reward_scheme=nn_rewards))

    """
    Agent 1
    """
    # agents.append(AStarAgent(1, args.goals[1]))
    # agents.append(RandomAgent(1, args.goals[1], p=0.8))
    # agents.append(DijkstraAgent(1, args.goals[1]))
    # agents.append(SafeAgent(1, args.goals[1]))
    # agents.append(POMDPAgent(1, args.goals[1], exist_policy=True))
    agents.append(MDPAgent(1, args.goals[1], belief_update=True, verbose=True))
    # agents.append(MetaAgent(1, args.goals[1], belief_update=True, verbose=False,
    #                         meta_policy=meta_square))  # `meta_small` or `meta_square`
    # agents.append(CBSAgent(1, args.goals[1], soft_update=2e-5, verbose=True))
    # agents.append(QMDPAgent(1, args.goals[1]))
    # agents.append(UniformTreeSearchAgent(1, args.goals[1],
    #                                      belief_update=True, depth=2, node_eval='HEU-C',
    #                                      verbose=False,
    #                                      check_repeated_states=True))
    # nn_rewards = {
    #     'illegal': 1,
    #     'normal': 1,
    #     'collision': 3000,
    #     'goal': 10
    # }
    # agents.append(UniformTreeSearchAgent(1, args.goals[1],
    #                                      belief_update=True,
    #                                      depth=2, node_eval='NN',
    #                                      verbose=True,
    #                                      nn_estimator=meta_small,
    #                                      reward_scheme=nn_rewards))
    # agents.append(UniformTreeSearchAgent(1, args.goals[1],
    #                                      belief_update=True,
    #                                      depth=2, node_eval='CBS',
    #                                      sample_eval=10,
    #                                      sample_backup=10,
    #                                      verbose=True))
    # agents.append(AsymmetricTreeSearch(1, args.goals[1],
    #                                    belief_update=True, verbose=True,
    #                                    max_it=3e2,
    #                                    node_eval='CBS',
    #                                    sample_eval=1,
    #                                    sample_select=50,
    #                                    pUCB=True))
    # agents.append(HistoryMDPAgent(1, args.goals[1], horizon=4))  # a legacy agent

    """
    The rest: assume A-star by default
    """
    for i in range(2, args.agents):
        agents.append(AStarAgent(i, args.goals[i]))

    game = MAPF(agents,
                args.starts[:args.agents],
                args.goals[:args.agents],
                args.map,
                stop_on=None)  # can be any agent id (non-negative integer)
    history, steps, collisions, stuck = game.run()

    print('\nDetailed trajectory:')
    show_hist(history)
    print(steps, collisions, stuck)

    if args.vis:
        paths = []
        for step in history:
            paths.append(step[1])
        if max(args.map.shape) in range(10):
            FPS = 60
        elif max(args.map.shape) in range(10, 15):
            FPS = 30
        else:
            FPS = 15
        animator = Animation(range(args.agents),
                             args.map,
                             args.starts[:args.agents],
                             args.goals[:args.agents],
                             paths,
                             FPS=FPS)
        animator.show()
        if args.save:
            animator.save(file_name=args.save, speed=100)
