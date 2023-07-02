from search_agent import AStarAgent, DijkstraAgent
from ma_env import MAPF
from animator import Animation

import argparse
import os

import numpy as np


def parse_map_from_file(map_config):
    PREFIX = 'maps/'
    POSTFIX = '.map'
    if not os.path.exists(PREFIX + map_config + POSTFIX):
        raise ValueError('Map config does not exist!')
    layout = []
    with open(PREFIX + map_config + POSTFIX, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('#'):
                pass
            else:
                row = []
                for char in line:
                    if char == '.':
                        row.append(0)
                    elif char == '@':
                        row.append(1)
                    else:
                        continue
                layout.append(row)
            line = f.readline()
    return np.array(layout)


def parse_locs(locs):
    loc_dict = dict()
    for i, l in enumerate(locs):
        loc_dict[f'p{i + 1}'] = eval(l.replace('_', ','))
    return loc_dict


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


def show_args(args):
    args = vars(args)
    for key in args:
        print(f'{key.upper()}:')
        print(args[key])
        print('-------------\n')


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
    agents.append(DijkstraAgent(1, args.goals[args.agents[1]]))

    game = MAPF(agents,
                list(args.starts.values()),
                list(args.goals.values()),
                args.map)
    history = game.run()
    show_hist(history)

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
