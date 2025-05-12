import random
from copy import deepcopy
from queue import PriorityQueue
from collections import namedtuple

import numpy as np

from utils import move, rev_action, hash, man_dist, get_avai_actions_mapf
from search_agent import astar, AStarAgent

np.random.seed(618)


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


class ChasingAgent(AStarAgent):
    """docstring for ChasingAgent"""

    def __init__(self, label, goal, p_chase=0.8):
        super(ChasingAgent, self).__init__(label, goal)
        self.p_chase = p_chase

    def act(self, state):
        astar_action = super(ChasingAgent, self).act(state)

        _, _, locations, layout = state
        if np.random.rand() > self.p_chase:
            return astar_action

        # Randomly select an opponent to chase
        rand_oppo_loc = random.choice(list(locations)[:self.label] + list(locations)[self.label + 1:])
        self.chasingplan = astar(locations[self.label], rand_oppo_loc, layout)
        if self.chasingplan:
            chasing_action = self.chasingplan[0]
        else:
            chasing_action = 0

        new_init = move(locations[self.label], chasing_action)
        self.plan = astar(new_init, self.goal, layout)
        self.round = 0

        return chasing_action


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


class EnhancedSafeAgent(SafeAgent):
    def act(self, state):
        N, prev_actions, locations, layout = state
        if locations[self.label] == self.goal:
            return 0

        best_safe_action = super(EnhancedSafeAgent, self).act(state)

        if getattr(self, 'prev_locations', None) and best_safe_action == 0:
            # check whether neighbor agents stay put
            prev_neighbors = []
            curr_neighbors = []
            for j, loc_j in enumerate(locations):
                if j == self.label:
                    continue
                if man_dist(locations[self.label], loc_j) <= 3:  # could be geq 2
                    curr_neighbors.append(loc_j)
                    prev_neighbors.append(self.prev_locations[j])

            if curr_neighbors == prev_neighbors:
                revised_layout = deepcopy(layout)
                for neighbor in curr_neighbors:
                    revised_layout[neighbor] = 1

                try:
                    self.plan = astar(locations[self.label], self.goal, revised_layout)
                    best_safe_action = self.plan[0]
                    self.round = 1
                except Exception:
                    pass

        self.prev_locations = locations

        return best_safe_action
