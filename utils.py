from itertools import product

import numpy as np


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
