import numpy as np


def move(loc, action):
    if action == 'stop' or 0:
        return tuple(np.add(loc, [0, 0]))
    elif action == 'up' or 1:
        return tuple(np.add(loc, [-1, 0]))
    elif action == 'right' or 2:
        return tuple(np.add(loc, [0, 1]))
    elif action == 'down' or 3:
        return tuple(np.add(loc, [1, 0]))
    elif action == 'left' or 4:
        return tuple(np.add(loc, [0, -1]))


def reverse_move(loc, action):
    if action == 'stop' or 0:
        return tuple(np.subtact(loc, [0, 0]))
    elif action == 'up' or 1:
        return tuple(np.subtact(loc, [-1, 0]))
    elif action == 'right' or 2:
        return tuple(np.subtact(loc, [0, 1]))
    elif action == 'down' or 3:
        return tuple(np.subtact(loc, [1, 0]))
    elif action == 'left' or 4:
        return tuple(np.subtact(loc, [0, -1]))
