from naive_agent import SafeAgent, RandomAgent
from search_agent import AStarAgent, DijkstraAgent
from mdp_agent import MDPAgent, HistoryMDPAgent
from pomdp_agent import POMDPAgent, QMDPAgent
from ts_agent import UniformTreeSearchAgent, AsymmetricTreeSearch
from ma_env import MAPF
from animator import Animation
from utils import parse_map_from_file, parse_locs, show_args

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent Planning.'
    )

    parser.add_argument('--agents', dest='agents', type=str, nargs='+',
                        help='Specify a team of agents,'
                             'e.g. p1 p2')
    parser.add_argument('--map', dest='map', type=str,
                        help='Specify a map')
    parser.add_argument('--starts', dest='starts', type=str, nargs='+',
                        help='Specify the starts for each agent,'
                             'e.g. 2_0 0_2')
    parser.add_argument('--goals', dest='goals', type=str, nargs='+',
                        help='Specify the goals for each agent,'
                             'e.g. 2_0 0_2')
    parser.add_argument('--vis', dest='vis', action='store_true',
                        help='Visulize the process')
    parser.add_argument('--save', dest='save', type=str,
                        help='Specify the path to save the animation')

    args = parser.parse_args()
    args.map = parse_map_from_file(args.map)
    args.starts = parse_locs(args.starts)
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
    agents.append(AStarAgent(0, args.goals[args.agents[0]]))
    # agents.append(SafeAgent(0, args.goals[args.agents[0]]))
    # agents.append(RandomAgent(0, args.goals[args.agents[0]]))
    # agents.append(DijkstraAgent(0, args.goals[args.agents[0]]))

    # agents.append(SafeAgent(1, args.goals[args.agents[1]]))
    # agents.append(POMDPAgent(1, args.goals[args.agents[1]], exist_policy=True))
    # agents.append(MDPAgent(1, args.goals[args.agents[1]], belief_update=False, verbose=True))
    # agents.append(QMDPAgent(1, args.goals[args.agents[1]]))
    # agents.append(HistoryMDPAgent(1, args.goals[args.agents[1]], horizon=4))
    agents.append(UniformTreeSearchAgent(1, args.goals[args.agents[1]],
                                         belief_update=True, depth=2, node_eval='HEU-C',
                                         check_repeated_states=True))
    # agents.append(AsymmetricTreeSearch(1, args.goals[args.agents[1]],
    #                                    belief_update=True, max_it=1e2))

    # agents.append(DijkstraAgent(2, args.goals[args.agents[2]]))

    game = MAPF(agents,
                list(args.starts.values()),
                list(args.goals.values()),
                args.map)
    history, steps = game.run()
    show_hist(history)
    print(steps)

    if args.vis:
        paths = []
        for step in history:
            paths.append(step[1])
        animator = Animation(args.agents,
                             args.map,
                             list(args.starts.values()),
                             list(args.goals.values()),
                             paths,
                             FPS=60)
        animator.show()
        if args.save:
            animator.save(file_name=args.save, speed=100)
