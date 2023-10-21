from itertools import product
import os

import numpy as np

"""
Commandline helper functions
"""


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


def show_args(args):
    args = vars(args)
    for key in args:
        print(f'{key.upper()}:')
        print(args[key])
        print('-------------\n')


"""
Board game helper functions
"""


def move(loc, action):
    if action == 'stop' or action == 0:
        return tuple(np.add(loc, [0, 0]))
    elif action == 'up' or action == 1:
        return tuple(np.add(loc, [-1, 0]))
    elif action == 'right' or action == 2:
        return tuple(np.add(loc, [0, 1]))
    elif action == 'down' or action == 3:
        return tuple(np.add(loc, [1, 0]))
    elif action == 'left' or action == 4:
        return tuple(np.add(loc, [0, -1]))


def reverse_move(loc, action):
    if action == 'stop' or action == 0:
        return tuple(np.subtact(loc, [0, 0]))
    elif action == 'up' or action == 1:
        return tuple(np.subtact(loc, [-1, 0]))
    elif action == 'right' or action == 2:
        return tuple(np.subtact(loc, [0, 1]))
    elif action == 'down' or action == 3:
        return tuple(np.subtact(loc, [1, 0]))
    elif action == 'left' or action == 4:
        return tuple(np.subtact(loc, [0, -1]))


def rev_action(action):
    """
    Returns the reversed action
    """
    if action == 0:
        return 0
    else:
        return (action - 1 + 2) % 4 + 1


def hash(num_content):
    return str(num_content)


def soft_max(x, mul=2):
    x = x * mul
    return np.exp(x) / np.sum(np.exp(x))


def enumerate_all(N, layout):
    """
    Enumerate the set of all env states, permutation sensitive.
    """
    nrows = len(layout)
    ncols = len(layout[0])

    def idx2row(idx):
        return idx // ncols

    def idx2col(idx):
        return idx % ncols

    all_states = []
    for idxs in product(range(nrows * ncols), repeat=N):
        state = []
        onwall = False
        for idx in idxs:
            if layout[(idx2row(idx), idx2col(idx))] == 1:
                onwall = True
                break
            state.append((idx2row(idx), idx2col(idx)))
        if onwall:
            continue
        all_states.append(tuple(state))

    return all_states


def euc_dist(loc, dest):
    return np.sqrt(np.sum(np.square(np.array(loc) - np.array(dest))))


def man_dist(loc, dest):
    return np.sum(np.abs(np.array(loc) - np.array(dest)))


def MSE(src, dest):
    return np.mean(np.square(np.array(src) - np.array(dest)))


"""
MAPF transition & reward for sing-agent MDP formulation
"""


def T_mapf(label, goal, layout, locations, action_profile):
    """
    The multi-agent env transition:
    given a tuple of locations and an action profile,
    returns the successor locations.
    """
    # If edge conflict, no valid transition
    if locations == 'EDGECONFLICT':
        return 'EDGECONFLICT'

    # If goal, no valid transition
    if locations[label] == goal:
        return tuple(locations)

    # If vertex conflict, no valid transition
    for i, other_loc in enumerate(locations):
        if i != label and other_loc == locations[label]:
            return tuple(locations)

    nrows = len(layout)
    ncols = len(layout[0])
    succ_locations = []
    for i in range(len(locations)):
        succ_loc = move(locations[i], action_profile[i])
        if layout[succ_loc] == 1 or\
                succ_loc[0] not in range(1, nrows + 1) or\
                succ_loc[1] not in range(1, ncols + 1):
            # Go into walls -> bounce back
            succ_locations.append(locations[i])
        else:
            succ_locations.append(succ_loc)

    # If adjacent swap, mark as edge conflict
    for i, other_loc in enumerate(succ_locations):
        if i != label:
            if other_loc == locations[label] and\
                    succ_locations[label] == locations[i]:
                return 'EDGECONFLICT'

    return tuple(succ_locations)


def R_mapf(label, goal, pred_locs, succ_locs, penalty=1e6):
    """
    The multi-agent env reward for this pivotal agent:
    given the prev and succ locations,
    returns the reward.
    """
    # Edge conflict
    if succ_locs == 'EDGECONFLICT':
        return -penalty

    # Vertex conflict
    for i, other_loc in enumerate(succ_locs):
        if i != label and other_loc == succ_locs[label]:
            return -penalty

    if succ_locs[label] == goal:
        return 1000
    return -1


def get_avai_actions_mapf(loc, layout):
    nrows = len(layout)
    ncols = len(layout[0])
    avai_actions = []
    for a in range(5):
        succ_loc = move(loc, a)
        if layout[succ_loc] == 1 or\
                succ_loc[0] not in range(1, nrows + 1) or\
                succ_loc[1] not in range(1, ncols + 1):
            continue
        avai_actions.append(a)
    return avai_actions
