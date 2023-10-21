from utils import move

from itertools import product
from collections import namedtuple
from copy import deepcopy

import numpy as np


State = namedtuple('State',
                   ['NumAgents', 'PrevActions', 'Locations', 'Layout'])


class MAPF(object):
    """docstring for MAPF"""

    def __init__(self, agents, starts, goals, layout):
        """
        agents: a list of agents
        starts/goals: a list of tuples
        layout: a 2d array; 0-empty, 1-obstacle
        """
        super(MAPF, self).__init__()
        self.agents = agents
        self.N = len(agents)
        self.starts = tuple(starts)
        self.goals = tuple(goals)
        self.layout = layout
        # self.all_states = self.enumerate_all()

        self.state = State(self.N, None, self.starts, self.layout)
        self.reach_goal = np.zeros(self.N)
        self.max_steps = max(self.layout.shape) * 10

    def transit(self, action_profile):
        """
        Given a state and an action profile,
        returns a successor state or a vector of prob.
        """
        N, prev_actions, locations, layout = self.state
        avai_actions = self.get_avai_actions()

        succ_locations = []
        for i in range(N):
            if action_profile[i] not in avai_actions[i]:
                raise RuntimeError(f"Action {action_profile[i]} not allowed in {locations}!")
            succ_loc = move(locations[i], action_profile[i])
            succ_locations.append(succ_loc)

        self.state = State(N, action_profile, tuple(succ_locations), layout)
        return self.state

    def get_avai_actions(self):
        """
        Given a state,
        returns feasible actions for each agent.
        Only forbids actions into walls or beyond the map
        """
        N, prev_actions, locations, layout = self.state
        nrows = len(self.layout)
        ncols = len(self.layout[0])

        avai_actions = []
        for i in range(N):
            loc = locations[i]
            actions = []
            for a in range(5):
                succ_loc = move(loc, a)
                if layout[succ_loc] == 1 or\
                        succ_loc[0] not in range(1, nrows + 1) or\
                        succ_loc[1] not in range(1, ncols + 1):
                    continue
                actions.append(a)
            avai_actions.append(actions)

        return avai_actions

    def check_end(self):
        """
        Given a state,
        returns whether it is an end.
        """
        _, _, locations, _ = self.state
        for i, loc in enumerate(locations):
            if loc != self.goals[i]:
                self.reach_goal[i] += 1
        if locations == self.goals:
            return True
        return False

    def run(self):
        """
        Returns history: a list of aux info at each state,
        In this env, returns a list of (actions, succ_locations)
        """
        history = [(None, self.state.Locations)]
        self.step = 0
        while not self.check_end():
            action_profile = []
            for i in range(self.N):
                action_profile.append(self.agents[i].act(self.state))
            _, action_profile, locations, _ = self.transit(action_profile)
            history.append((deepcopy(action_profile), locations))
            self.step += 1
            if self.step > self.max_steps:
                break
        return history, self.reach_goal
