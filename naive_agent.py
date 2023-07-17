from utils import move, rev_action, hash, man_dist, get_avai_actions_mapf
from search_agent import astar, AStarAgent

from queue import PriorityQueue
from collections import namedtuple

import numpy as np


class RandomAgent(AStarAgent):
    """docstring for RandomAgent"""

    def __init__(self, label, goal, p=0.7):
        super(RandomAgent, self).__init__(label, goal)
        self.p = p

    def act(self, state):
        astar_action = super(RandomAgent, self).act(state)

        _, _, locations, layout = state
        avai_actions = get_avai_actions_mapf(locations[self.label], layout)
        if np.random.rand() < self.p:
            return astar_action

        rand_action = np.random.choice(avai_actions)

        # Restore the plan
        new_init = move(locations[self.label], rand_action)
        self.plan = astar(new_init, self.goal, layout)
        self.round = 0

        return rand_action


class SafeAgent(AStarAgent):
    """docstring for SafeAgent"""

    def act(self, state):
        astar_action = super(SafeAgent, self).act(state)

        _, _, locations, layout = state
        avai_actions = get_avai_actions_mapf(locations[self.label], layout)
        safe_actions = []
        for a in avai_actions:
            succ_loc = move(locations[self.label], a)
            is_safe = True
            for op_id, op_loc in enumerate(locations):
                if op_id == self.label:
                    continue
                if man_dist(succ_loc, op_loc) <= 1:
                    is_safe = False
                    break
            if is_safe:
                safe_actions.append(a)
        if astar_action in safe_actions:
            return astar_action

        best_safe_action = 0  # If no safe action, then stop by default
        best_safe_action_dist = 9999
        for a in safe_actions:
            if man_dist(move(locations[self.label], a),
                        self.goal) < best_safe_action_dist:
                best_safe_action = a
                best_safe_action_dist = man_dist(move(locations[self.label], a),
                                                 self.goal)

        # Restore the plan
        new_init = move(locations[self.label], best_safe_action)
        self.plan = astar(new_init, self.goal, layout)
        self.round = 0

        return best_safe_action
