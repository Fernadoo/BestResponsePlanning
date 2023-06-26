from utils import move

from queue import PriorityQueue
from collections import namedtuple

import numpy as np


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

    def astar(self, init, layout):
        """
        Ignore the others, simply do astar search,
        and always stick to the plan, never replan
        """
        Node = namedtuple('Node',
                          ['fValue', 'gValue', 'PrevAction', 'Loc'])
        nrows = len(layout)
        ncols = len(layout[0])

        def euc_heu(loc):
            return np.sqrt(np.sum(np.square(loc - self.goal)))

        def man_heu(loc):
            return np.sum(np.abs(loc - self.goal))

        def get_successors(node):
            f, g, prev_action, curr_loc = node
            successors = []
            for a in range(5):
                succ_loc = move(curr_loc, a)
                if layout[succ_loc] == 1 or\
                        succ_loc[0] not in range(nrows) or\
                        succ_loc[1] not in range(ncols):
                    continue
                heu = euc_heu(succ_loc)
                succ_node = Node(heu + g + 1, g + 1, a, succ_loc)
                successors.append(succ_node)
            return successors

        plan = []
        visited = []
        parent_dict = dict()
        q = PriorityQueue()
        q.put(Node(euc_heu(init), 0, None, init))
        while not q.empty():
            curr_node = q.get()
            if curr_node.Loc == self.goal:
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

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.plan is None:
            self.plan = self.astar(locations[self.label], layout)

        action = self.plan[self.round]
        self.round += 1
        return action
