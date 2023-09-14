from search_agent import dijkstra
from utils import (move, hash, soft_max, enumerate_all, MSE,
                   T_mapf, R_mapf, get_avai_actions_mapf)

import os
from collections import namedtuple
from itertools import product
from copy import deepcopy
import time
import pickle

import numpy as np
from mdptoolbox import mdp


class MDPAgent(object):
    """
    MDP agent:
    [Assuming the belief will NOT change in the futher]
    1. Initialize the belief about others
    2. For each iteration:
        i) Construct the MDP induced by the belief
        ii) Do the next one step as the opt policy indicates
    3. Observe how the others played and update the belief
    """

    def __init__(self, label, goal, belief_update=True):
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

    def update_belief(self, beliefs, prev_locs, prev_actions, soft=1e-2):
        beliefs_pi, beliefs_prob = beliefs
        new_beliefs_prob = deepcopy(beliefs_prob)
        # prev_locs = self.prev_locations

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

            # Soft-update
            # since some action may not be included in any support policy
            new_probs += soft
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
                Sj = T_mapf(self.label, self.goal, self.layout, Si, actions)
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
                Sj = T_mapf(self.label, self.goal, self.layout, Si, actions)
                j = S.index(Sj)
                A = actions[self.label]
                reward = R_mapf(self.label, self.goal, Si, Sj, penalty=1e3)

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
        # For theory proving
        with open('mdp_values.pkl', 'wb') as pklf:
            pickle.dump((self.label, self.goal, self.layout,
                         S, VI.V, VI.policy),
                        pklf)
        return VI.policy

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
            self.beliefs = self.update_belief(self.beliefs, self.prev_locations, prev_actions)
            self.policy = self.translate_solve()

        self.print_belief(self.beliefs)
        Si = self.S.index(locations)
        action = self.policy[Si]
        self.prev_locations = locations
        return action

    def print_belief(self, beliefs):
        beliefs_prob = beliefs[1]
        for i, probs in enumerate(beliefs_prob):
            if i == self.label:
                continue
            idx = 0
            belief_map = np.zeros(shape=self.layout.shape)
            for r in range(len(self.layout)):
                for c in range(len(self.layout[0])):
                    if self.layout[r, c] == 1 or (r, c) == self.goal:
                        belief_map[r, c] = np.nan
                    else:
                        belief_map[r, c] = np.round(probs[idx], 3)
                        idx += 1
            print(f'=== Belief({self.label + 1} -> {i + 1}) ===')
            print(belief_map)
        print()


class HistoryMDPAgent(MDPAgent):
    """
    History-MDP Agent:
    [Coupling belief revision into transition modelling]
    State := (Loc-tuple, History)
    History := [(Loc-tuple, joint_a), ...]
    BeliefRvision: B x H -> B
    T[S2=(s2,h2) | S1=(s1,h1), a] := T_{env}[s2 | s1, a] [+] BeliefRevision(b, h1)
    """

    def __init__(self, label, goal, belief_update=True, horizon=3, err=1e-2):
        super(HistoryMDPAgent, self).__init__(label, goal, belief_update)
        self.horizon = horizon
        self.err = err

    def act(self, state):
        t1 = time.time()
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # Formulate a history-MDP based on the belief
        if self.policy is None:
            self.policy = self.translate_solve(locations)

        # Update the belief and replan
        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(self.beliefs, self.prev_locations, prev_actions, soft=1e-10)
            self.policy = self.translate_solve(locations)

        self.print_belief(self.beliefs)

        curr_h_state = (locations, [])
        action = np.argmax(self.policy[hash(curr_h_state)])
        if action not in get_avai_actions_mapf(locations[self.label], self.layout):
            action = 0
        t2 = time.time()
        print(action, f'took {t2-t1}s')
        self.prev_locations = locations
        return action

    def translate_solve(self, curr_locs):
        """
        Formulate a history-MDP from the pivotal agent's perspective,
        with initial h_state as (curr_locs, []),
        and do value iteration for finite times or until the error is small enough
        """
        if getattr(self, 'beliefs', None) is None or\
                getattr(self, 'layout', None) is None:
            raise RuntimeError("Get invalid beliefs, "
                               "or invalid layout!")

        # Get all possible states
        if getattr(self, 'S', None) is None:
            self.S = enumerate_all(self.num_agents, self.layout)
            self.S.append('EDGECONFLICT')
            self.num_all = len(self.S)

        init_h_state = (curr_locs, [])
        h_states = [init_h_state]
        h_qvalues = {hash(init_h_state): np.zeros(5)}
        it = 0
        mse = np.inf
        while it < self.horizon or mse < self.err:
            h_states, h_qvalues = self.value_iteration(h_states, h_qvalues)
            it += 1

        return h_qvalues

    def value_iteration(self, h_states, h_qvalues, gamma=0.9):
        new_h_states = deepcopy(h_states)
        new_h_qvalues = deepcopy(h_qvalues)
        beliefs_pi, beliefs_prob = self.beliefs
        beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
        beliefs_num[self.label] = range(1)

        # Bellman optimality backup
        # Q(s,a) = \sum_{s'} T(s'|s,a) [R(s, a, s') + \gamma \max Q(s',a)]
        for i in range(len(h_states) - 1, -1, -1):  # reversed traverse
            hs = hash(h_states[i])
            locs, h = eval(hs)
            _, beliefs_prob = self.belief_revision(self.beliefs, h)

            for a in range(5):
                T_a = []
                R_a = []
                succ_hs = []

                if locs == 'EDGECONFLICT':
                    succ_locs = T_mapf(self.label, self.goal, self.layout, locs, None)
                    reward = R_mapf(self.label, self.goal, locs, succ_locs)
                    T_a.append(1)
                    R_a.append(reward)
                    succ_hs.append((succ_locs, 'EOH'))  # End of history

                else:
                    for other_joint_a in product(range(5), repeat=self.num_agents - 1):
                        joint_a = list(other_joint_a)
                        joint_a.insert(self.label, a)

                        succ_locs = T_mapf(self.label, self.goal, self.layout, locs, joint_a)
                        reward = R_mapf(self.label, self.goal, locs, succ_locs)

                        prob = 0
                        r = 0
                        for joint_Pi in product(*beliefs_num):
                            coef = 1
                            for op_id, pi_id in enumerate(joint_Pi):
                                if op_id == self.label:
                                    continue
                                coef *= (
                                    beliefs_pi[op_id][pi_id][hash(locs[op_id])][joint_a[op_id]]
                                    * beliefs_prob[op_id][pi_id]
                                )
                            prob += coef
                            r += coef * reward

                        T_a.append(prob)
                        R_a.append(r)

                        succ_h = deepcopy(h)
                        succ_h.append((locs, joint_a))
                        succ_hs.append((succ_locs, succ_h))

                # Normalize transitions
                T_a = T_a / np.sum(T_a)

                # Bellman backup over growing states
                new_h_qvalues[hs][a] = 0
                for j, sj in enumerate(succ_hs):
                    next_V = 0
                    if hash(sj) not in h_qvalues:
                        new_h_qvalues[hash(sj)] = np.zeros(5)
                        new_h_states.append(sj)  # create new h_state
                    else:
                        next_V = np.max(new_h_qvalues[hash(sj)])  # In-place update
                    new_h_qvalues[hs][a] += T_a[j] * (R_a[j] + gamma * next_V)

        return new_h_states, new_h_qvalues

    def belief_revision(self, beliefs, history):
        if history == 'EOH':
            return None, None
        for prev_locs, prev_actions in history:
            beliefs = self.update_belief(beliefs, prev_locs, prev_actions, soft=1e-10)
        return beliefs
