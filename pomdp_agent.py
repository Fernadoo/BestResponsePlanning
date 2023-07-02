import os
from collections import namedtuple


class POMDPAgent(object):
    """docstring for POMDPAgent"""

    def __init__(self, label, goal):
        super(POMDPAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.plan = None
        self.round = 0
        self.history = []

        # beliefs := [belief_i]
        # belief_i := policy -> pr
        # policy := state -> action_dist
        self.beliefs = self.init_belief()

    def init_belief(self):
        return

    def update_belief(self, pred_state, pred_action):
        return

    def translate_solve(self,):
        """
        Invoke pomdp-solver
        """
        return

    def replan(self,):
        return

    def act(self, state):
        N, prev_actions, locations, layout = state

        return
