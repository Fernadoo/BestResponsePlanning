from search_agent import dijkstra
from utils import move, hash, soft_max, enumerate_all

import os
from collections import namedtuple
from itertools import product
from copy import deepcopy

import numpy as np
from mdptoolbox import mdp


class MDPAgent(object):
    """docstring for MDPAgent"""

    def __init__(self, label, goal, belief_update=False):
        super(MDPAgent, self).__init__()
        self.label = label
        self.goal = goal
        self.history = []

        # beliefs := [belief_i]
        # belief_i := policy -> pr
        # policy := state -> action_dist
        self.beliefs = None
        self.policy = None
        self.belief_update = belief_update

    def init_belief(self, num_agents, layout):
        """
        beliefs := [belief_i]
        belief_i := policy -> pr
        policy := state -> action_dist
        """
        beliefs_pi = [[] for i in range(num_agents)]
        beliefs_prob = [[] for i in range(num_agents)]
        NORM_PR = 1

        # Enumerate all possible goals except her own one
        feasible_goals = []
        for r in range(len(layout)):
            for c in range(len(layout[0])):
                if layout[r, c] == 0 and (r, c) != self.goal:
                    feasible_goals.append((r, c))

        # For every other agents, for every possible goals
        # find the optimal single-agent policies (shortest-path tree)
        for i in range(num_agents):
            if i == self.label:
                continue
            for goal in feasible_goals:
                policy = dijkstra(goal, layout)
                for loc in policy:
                    action_dist = [0, 0, 0, 0, 0]
                    action_dist[policy[loc]] = 1
                    # action_dist = soft_max(action_dist)
                    policy[loc] = action_dist
                beliefs_pi[i].append(policy)
                beliefs_prob[i].append(NORM_PR)
            beliefs_prob[i] = beliefs_prob[i] / np.sum(beliefs_prob[i])

        return beliefs_pi, beliefs_prob

    def update_belief(self, prev_actions):
        beliefs_pi, beliefs_prob = self.beliefs
        new_beliefs_prob = deepcopy(beliefs_prob)
        prev_locs = self.prev_locations

        # Bayesian update:
        # B_pi_i^j \prop pi_i^j[Si][ai] * pi_i^j
        for i, Pi_i in enumerate(beliefs_pi):
            if i == self.label:
                continue
            new_probs = np.zeros(len(Pi_i))
            for j, pi in enumerate(Pi_i):
                new_probs[j] = (
                    Pi_i[j][hash(prev_locs[i])][prev_actions[i]]
                    * beliefs_prob[i][j]
                )
            # soft-update
            new_probs += 0.01
            new_beliefs_prob[i] = new_probs / np.sum(new_probs)

        return beliefs_pi, new_beliefs_prob

    def translate_solve(self):
        """
        Formulate an MDP from the pivotal agent's perspective,
        and invoke mdptoolbox
        """
        if getattr(self, 'beliefs', None) is None or\
                getattr(self, 'layout', None) is None:
            raise RuntimeError("Get invalid beliefs, "
                               "or invalid layout!")

        layout = self.layout
        nrows = len(layout)
        ncols = len(layout[0])

        def T_ma(locations, action_profile):
            """
            The multi-agent env transition:
            given a tuple of locations and an action profile,
            returns the successor locations.
            """
            # If edge conflict, no valid transition
            if locations == 'EDGECONFLICT':
                return 'EDGECONFLICT'

            # If goal, no valid transition
            if locations[self.label] == self.goal:
                return tuple(locations)

            # If vertex conflict, no valid transition
            for i, other_loc in enumerate(locations):
                if i != self.label and other_loc == locations[self.label]:
                    return tuple(locations)

            succ_locations = []
            for i in range(self.num_agents):
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
                if i != self.label:
                    if other_loc == locations[self.label] and\
                            succ_locations[self.label] == locations[i]:
                        return 'EDGECONFLICT'

            return tuple(succ_locations)

        def R_ma(pred_locs, succ_locs):
            """
            The multi-agent env reward for this pivotal agent:
            given the prev and succ locations,
            returns the reward.
            """
            # Edge conflict
            if succ_locs == 'EDGECONFLICT':
                return -1000

            # Vertex conflict
            for i, other_loc in enumerate(succ_locs):
                if i != self.label and other_loc == succ_locs[self.label]:
                    return -1000

            if succ_locs[self.label] == self.goal:
                return 1000
            return -1

        # Get all possible states
        if getattr(self, 'S', None) is None:
            self.S = enumerate_all(self.num_agents, self.layout)
            self.S.append('EDGECONFLICT')
            self.num_all = len(self.S)

        S = self.S
        num_all = self.num_all
        beliefs_pi, beliefs_prob = self.beliefs
        beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
        beliefs_num[self.label] = range(1)

        # Translate transition matrix shape(T) := (A,S,S)
        T = np.zeros(shape=(5, num_all, num_all))
        for i, Si in enumerate(S):
            for actions in product(range(5), repeat=self.num_agents):
                Sj = T_ma(Si, actions)
                j = S.index(Sj)
                A = actions[self.label]

                # Starts with an edge conflict, always ends the same
                if Si == 'EDGECONFLICT':
                    T[A, i, j] = 1
                    continue

                # Treat all the others as transition noises
                for joint_idxs in product(*beliefs_num):
                    noise = 1
                    for op_id, pi_id in enumerate(joint_idxs):
                        if op_id == self.label:
                            continue
                        noise *= (
                            beliefs_pi[op_id][pi_id][hash(Si[op_id])][actions[op_id]]
                            * beliefs_prob[op_id][pi_id]
                        )
                    T[A, i, j] += noise
        # T is conceptually a stochastic matrix already,
        # But due to float ops, we need to further normalize it
        T = T / np.sum(T, axis=2).reshape(5, num_all, 1)

        # Translate reward matrix shape(R) := (A,S,S)
        R = np.zeros(shape=(5, num_all, num_all))
        for i, Si in enumerate(S):
            for actions in product(range(5), repeat=self.num_agents):
                Sj = T_ma(Si, actions)
                j = S.index(Sj)
                A = actions[self.label]
                reward = R_ma(Si, Sj)

                # Only cares about the landing state
                if Sj == 'EDGECONFLICT':
                    R[A, i, j] = reward
                    continue

                for joint_idxs in product(*beliefs_num):
                    coef = 1
                    for op_id, pi_id in enumerate(joint_idxs):
                        if op_id == self.label:
                            continue
                        coef *= (
                            beliefs_pi[op_id][pi_id][hash(Si[op_id])][actions[op_id]]
                            * beliefs_prob[op_id][pi_id]
                        )
                    R[A, i, j] += reward * coef

        VI = mdp.ValueIteration(T, R, discount=0.9)
        VI.run()
        return VI.policy

    def replan(self,):
        return

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # Formulate an MDP based on the belief
        # Alt1: Compute once at the beginning
        if self.policy is None:
            self.policy = self.translate_solve()

        # Alt2: Update the belief and replan
        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(prev_actions)
            self.policy = self.translate_solve()

        self.print_belief(self.beliefs[1][0])
        Si = self.S.index(locations)
        action = self.policy[Si]
        self.prev_locations = locations
        return action

    def print_belief(self, probs):
        idx = 0
        belief_map = np.zeros(shape=self.layout.shape)
        for r in range(len(self.layout)):
            for c in range(len(self.layout[0])):
                if self.layout[r, c] == 1 or (r, c) == self.goal:
                    belief_map[r, c] = np.nan
                else:
                    belief_map[r, c] = np.round(probs[idx], 3)
                    idx += 1
        print(belief_map, '\n')
