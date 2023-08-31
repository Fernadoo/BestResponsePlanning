from utils import move, rev_action, hash, euc_dist

from queue import PriorityQueue
from collections import namedtuple

import numpy as np


"""
A-STAR SEARCH
Inputs: init, goal, layout
Returns: a sequence (list) of actions
"""


def astar(init, goal, layout):
    """
    Ignore the others, simply do astar search,
    and always stick to the plan, never replan
    """
    Node = namedtuple('ANode',
                      ['fValue', 'gValue', 'PrevAction', 'Loc'])
    nrows = len(layout)
    ncols = len(layout[0])

    def get_successors(node):
        f, g, prev_action, curr_loc = node
        successors = []
        for a in range(5):
            succ_loc = move(curr_loc, a)
            if layout[succ_loc] == 1 or\
                    succ_loc[0] not in range(1, nrows + 1) or\
                    succ_loc[1] not in range(1, ncols + 1):
                continue
            heu = euc_dist(succ_loc, goal)
            succ_node = Node(heu + g + 1, g + 1, a, succ_loc)
            successors.append(succ_node)
        return successors

    plan = []
    visited = []
    parent_dict = dict()
    q = PriorityQueue()
    q.put(Node(euc_dist(init, goal), 0, None, init))
    while not q.empty():
        curr_node = q.get()
        if curr_node.Loc == goal:
            # backtrack to get the plan
            curr = curr_node
            while curr.Loc != init:
                plan.insert(0, curr.PrevAction)
                curr = parent_dict[curr]
            return plan

        if curr_node.Loc in visited:
            continue
        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
            parent_dict[succ_node] = curr_node
        visited.append(curr_node.Loc)
    raise RuntimeError("No astar plan found!")


class AStarAgent(object):
    """docstring for AStarAgent"""

    def __init__(self, label, goal):
        """
        label: an integer name
        """
        super(AStarAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.plan = None
        self.round = 0

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.plan is None:
            self.plan = astar(locations[self.label], self.goal, layout)
        if locations[self.label] == self.goal:
            return 0

        action = self.plan[self.round]
        self.round += 1
        return action


"""
DIJKSTRA SEARCH
Inputs: init, layout
Outputs: a policy (loc -> action) = shortest path tree from every goal to init
"""


def dijkstra(init, layout):
    Node = namedtuple('DNode',
                      ['gValue', 'PrevAction', 'Loc'])
    nrows = len(layout)
    ncols = len(layout[0])
    num_empty = nrows * ncols - np.sum(layout)

    def get_successors(node):
        g, prev_action, curr_loc = node
        successors = []
        for a in range(5):
            succ_loc = move(curr_loc, a)
            if layout[succ_loc] == 1 or\
                    succ_loc[0] not in range(1, nrows + 1) or\
                    succ_loc[1] not in range(1, ncols + 1):
                continue
            succ_node = Node(g + 1, a, succ_loc)
            successors.append(succ_node)
        return successors

    visited = []
    policy = dict()
    curr_num_visited = 0
    q = PriorityQueue()
    q.put(Node(0, 0, init))
    while not q.empty() and curr_num_visited < num_empty:
        curr_node = q.get()
        if curr_node.Loc in visited:
            continue
        curr_num_visited += 1
        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
        visited.append(curr_node.Loc)
        policy[hash(curr_node.Loc)] = rev_action(curr_node.PrevAction)
    if curr_num_visited < num_empty:
        raise RuntimeError("No dijkstra plan found!")
    return policy


class DijkstraAgent(object):
    """docstring for DijkstraAgent"""

    def __init__(self, label, goal):
        super(DijkstraAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.policy = None

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.policy is None:
            self.policy = dijkstra(self.goal, layout)
            # for loc in self.policy:
            #     print(loc, self.policy[loc])
        if locations[self.label] == self.goal:
            return 0

        action = self.policy[hash(locations[self.label])]
        return action
