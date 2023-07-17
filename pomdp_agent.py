from mdp_agent import T_mapf, R_mapf, get_avai_actions_mapf, MDPAgent
from search_agent import dijkstra
from utils import move, hash, soft_max, enumerate_all

import os
from collections import namedtuple
from itertools import product
from copy import deepcopy

import numpy as np
from mdptoolbox import mdp
from tqdm import tqdm


class POMDPAgent(MDPAgent):
    """
    POMDP agent:
    [Also planning for future belief update,
     cannot plan for infinite horizons,
     currently 3-horizon by default.]
    1. Initialize the belief about others
    2. For each iteration:
        i) Construct the POMDP induced by all others' possible policies
        ii) Do the next one step as the opt policy indicates
    3. Observe how the others played and update the belief
    """

    def __init__(self, label, goal, belief_update=True, exist_policy=False):
        super(POMDPAgent, self).__init__(label, goal, belief_update)
        self.exist_policy = exist_policy

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # First iteration
        if self.policy is None:
            self.policy = self.translate_solve()

        # Second iteration onwards
        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(prev_actions)
            # No need for a new pomdp policy
            # since it is already a mapping from beliefs to actions
            # self.policy = self.translate_solve()

        self.print_belief(self.beliefs)

        # policy (a finite state automaton) execution
        node2action, node2action_value_mat, obs2node_mat = self.policy
        _, beliefs_prob = self.beliefs
        state_dist = []
        for Si in self.S:
            Oi, joint_idxs = Si
            # TODO: handle edge conflict explicitly?
            if Oi != locations:
                state_dist.append(0)
                continue
            prob = 1
            for op_id, pi_id in enumerate(joint_idxs):
                if op_id == self.label:
                    continue
                prob *= beliefs_prob[op_id][pi_id]
            state_dist.append(prob)
        # Normalization due float op issues
        state_dist = state_dist / np.sum(state_dist)
        node_values = node2action_value_mat @ state_dist
        best_node = np.argmax(node_values)
        action = node2action[best_node]
        if action not in get_avai_actions_mapf(locations[self.label], self.layout):
            action = 0

        self.prev_locations = locations
        return action

    def translate_solve(self):
        """
        Formulate a POMDP from the pivotal agent's perspective,
        and invoke pomdp-solve
        """
        if getattr(self, 'beliefs', None) is None or\
                getattr(self, 'layout', None) is None:
            raise RuntimeError("Get invalid beliefs, "
                               "or invalid layout!")

        beliefs_pi, beliefs_prob = self.beliefs
        beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
        beliefs_num[self.label] = range(1)

        # Get all possible observations and states
        if getattr(self, 'Omega', None) is None:
            self.Omega = enumerate_all(self.num_agents, self.layout)
            self.Omega.append('EDGECONFLICT')
            self.num_all_obs = len(self.Omega)

            self.S = []
            for obs in self.Omega:
                for joint_idxs in product(*beliefs_num):
                    state = (obs, joint_idxs)
                    self.S.append(state)
            self.num_all_s = len(self.S)

        if self.exist_policy:
            pomdp = self.write_pomdp(None, None, None)
            policy = self.solve_pomdp(pomdp)
            return policy

        Omega = self.Omega
        num_all_obs = self.num_all_obs
        S = self.S
        num_all_s = self.num_all_s

        # Translate the state transition matrix
        # T: <action> -> (start_S, end_S)
        T = np.zeros(shape=(5, num_all_s, num_all_s))
        for i, Si in enumerate(S):
            Oi, joint_idxs = Si
            for joint_a in product(range(5), repeat=self.num_agents):
                Oj = T_mapf(self.label, self.goal, self.layout, Oi, joint_a)
                A = joint_a[self.label]
                Sj = (Oj, joint_idxs)
                j = S.index(Sj)

                # Starts with an edge conflict, always ends the same
                if Oi == 'EDGECONFLICT':
                    T[A, i, j] = 1
                    continue

                prob = 1
                for op_id, pi_id in enumerate(joint_idxs):
                    if op_id == self.label:
                        continue
                    prob *= beliefs_pi[op_id][pi_id][hash(Oi[op_id])][joint_a[op_id]]
                T[A, i, j] += prob
        # T is conceptually a stochastic matrix already,
        # But due to float ops, we need to further normalize it
        T = T / np.sum(T, axis=2).reshape(5, num_all_s, 1)

        # Translate the observation matrix
        # O: <action> -> (end_S, obs)
        Obs = np.zeros(shape=(5, num_all_s, num_all_obs))
        for a in range(5):
            for i, Si in enumerate(S):
                Oi, _ = Si
                j = Omega.index(Oi)
                Obs[a, i, j] = 1

        # Translate the reward matrix
        # R: <action>, <start_S> -> (end_S, obs)
        R = np.zeros(shape=(5, num_all_s, num_all_s, num_all_obs))
        for i_s, Si in enumerate(S):
            Oi, joint_idxs = Si
            for joint_a in product(range(5), repeat=self.num_agents):
                Oj = T_mapf(self.label, self.goal, self.layout, Oi, joint_a)
                A = joint_a[self.label]
                Sj = (Oj, joint_idxs)
                j_o = Omega.index(Oj)
                j_s = S.index(Sj)
                reward = R_mapf(self.label, self.goal, Oi, Oj)

                # Only cares about the landing state
                if Oj == 'EDGECONFLICT':
                    R[A, i_s, j_s, j_o] = reward
                    continue

                coef = 1
                for op_id, pi_id in enumerate(joint_idxs):
                    if op_id == self.label:
                        continue
                    coef *= beliefs_pi[op_id][pi_id][hash(Oi[op_id])][joint_a[op_id]]
                R[A, i_s, j_s, j_o] += coef * reward

        pomdp = self.write_pomdp(T, Obs, R)
        policy = self.solve_pomdp(pomdp)
        return policy

    def write_pomdp(self, T, Obs, R, discount=0.95):
        # Write a pomdp file
        pomdp_file_name = f'pomdp-solve/problems/{self.label}.POMDP'
        if self.exist_policy:
            return pomdp_file_name
        pomdp_file = open(pomdp_file_name, 'w')

        # initial belief state
        # currently omitted

        # Write preamable
        pomdp_file.write(
            f'discount: {discount}\n'
            f'values: reward\n'
            f'states: {self.num_all_s}\n'
            f'actions: 5\n'
            f'observations: {self.num_all_obs}\n\n'
        )

        # Write T
        for a in range(5):
            pomdp_file.write(f'T: {a}\n')
            for Si in range(self.num_all_s):
                row_i = ' '.join(map(lambda x: str(x), T[a, Si]))
                pomdp_file.write(f'{row_i}\n')
            pomdp_file.write('\n')

        # Write Obs
        for a in range(5):
            pomdp_file.write(f'O: {a}\n')
            for Si in range(self.num_all_s):
                row_i = ' '.join(map(lambda x: str(x), Obs[a, Si]))
                pomdp_file.write(f'{row_i}\n')
            pomdp_file.write('\n')

        # Write R
        for a in range(5):
            for Si in range(self.num_all_s):
                pomdp_file.write(f'R: {a} : {Si}\n')
                for Sj in range(self.num_all_s):
                    row_j = ' '.join(map(lambda x: str(x), R[a, Si, Sj]))
                    pomdp_file.write(f'{row_j}\n')
                pomdp_file.write('\n')

        pomdp_file.close()
        return pomdp_file_name

    def solve_pomdp(self, pomdp_file_name, h=3):
        sol_file_prefix = f'{pomdp_file_name[:-6]}_sol'
        if not self.exist_policy:
            # Solve the pomdp
            solver = './pomdp-solve/pomdp-solve-os-x.bin'
            os.system(f'{solver} -horizon {h} -inc_prune restricted_region '
                      f' -pomdp {pomdp_file_name} -o {sol_file_prefix}')

        # Parse the policy graph
        def str2num(s):
            if s == '-':
                return np.nan
            else:
                return eval(s)

        alpha = open(f'{sol_file_prefix}.alpha', 'r')
        node2action = []
        node2action_value_mat = []
        line = alpha.readline()
        while line:
            tokens = line.split()
            if len(tokens) == 0:
                line = alpha.readline()
                continue
            else:
                node2action.append(eval(tokens[0]))
                line = alpha.readline()
                values = list(map(str2num, line.split()))
                node2action_value_mat.append(values)
                line = alpha.readline()

        pg = open(f'{sol_file_prefix}.pg', 'r')
        obs2node_mat = []
        line = pg.readline()
        while line:
            tokens = line.split()
            if len(tokens) == 0:
                line = pg.readline()
                continue
            else:
                obs2node = list(map(str2num, tokens[2:]))
                obs2node_mat.append(obs2node)
                line = pg.readline()

        return (np.array(node2action),
                np.array(node2action_value_mat),
                np.array(obs2node_mat))


class QMDPAgent(POMDPAgent):
    """
    QMDP agent:
    [An approximation for POMDP agent]
    1. Inherit the POMDP formulation from the above
    2. Solve the underlying MDP first, and then for each belief state
        pi(B) = argmax_a SUM B(s) * Q(S,a)
    """

    def __init__(self, label, goal, belief_update=True):
        super(QMDPAgent, self).__init__(label, goal, belief_update)

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # First iteration
        # Solve for all underlying MDPs at the beginning
        if self.policy is None:
            self.policy = self.translate_solve()

        # From second iteration onwards,
        # Update the belief and mix the MDP policy,
        # No need for replanning
        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(prev_actions)

        self.print_belief(self.beliefs)

        # policy execution: maximize mixed Q values
        Q_mdp = self.policy
        i = self.Omega.index(locations)
        Qs = Q_mdp[:, i, :]

        _, beliefs_prob = self.beliefs
        probs = np.zeros(self.num_all_joint_pi)
        for idx, joint_idxs in enumerate(self.joint_Pi):
            pr = 1
            for op_id, pi_id in enumerate(joint_idxs):
                if op_id == self.label:
                    continue
                pr *= beliefs_prob[op_id][pi_id]
            probs[idx] = pr
        mixed_Q = probs @ Qs
        action = np.argmax(mixed_Q)

        # if action not in get_avai_actions_mapf(locations[self.label], self.layout):
        #     action = 0

        self.prev_locations = locations
        return action

    def translate_solve(self):
        """
        Formulate a POMDP from the pivotal agent's perspective,
        and invoke pomdp-solve
        """
        if getattr(self, 'beliefs', None) is None or\
                getattr(self, 'layout', None) is None:
            raise RuntimeError("Get invalid beliefs, "
                               "or invalid layout!")

        beliefs_pi, beliefs_prob = self.beliefs
        beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
        beliefs_num[self.label] = range(1)

        # Get all possible observations and states
        if getattr(self, 'Omega', None) is None:
            self.Omega = enumerate_all(self.num_agents, self.layout)
            self.Omega.append('EDGECONFLICT')
            self.num_all_obs = len(self.Omega)

            self.joint_Pi = list(product(*beliefs_num))
            self.num_all_joint_pi = len(self.joint_Pi)

        Omega = self.Omega
        num_all_obs = self.num_all_obs
        joint_Pi = self.joint_Pi
        num_all_joint_pi = self.num_all_joint_pi

        # Translate the state transition matrix
        T = np.zeros(shape=(num_all_joint_pi, 5, num_all_obs, num_all_obs))
        for idx, joint_idxs in enumerate(joint_Pi):
            for i, Oi in enumerate(Omega):
                for joint_a in product(range(5), repeat=self.num_agents):
                    Oj = T_mapf(self.label, self.goal, self.layout, Oi, joint_a)
                    A = joint_a[self.label]
                    j = Omega.index(Oj)

                    # Starts with an edge conflict, always ends the same
                    if Oi == 'EDGECONFLICT':
                        T[idx, A, i, j] = 1
                        continue

                    prob = 1
                    for op_id, pi_id in enumerate(joint_idxs):
                        if op_id == self.label:
                            continue
                        prob *= beliefs_pi[op_id][pi_id][hash(Oi[op_id])][joint_a[op_id]]
                    T[idx, A, i, j] += prob
        # T is conceptually a stochastic matrix already,
        # But due to float ops, we need to further normalize it
        T = T / np.sum(T, axis=3).reshape(num_all_joint_pi, 5, num_all_obs, 1)

        # Translate the reward matrix
        R = np.zeros(shape=(num_all_joint_pi, 5, num_all_obs, num_all_obs))
        for idx, joint_idxs in enumerate(joint_Pi):
            for i, Oi in enumerate(Omega):
                for joint_a in product(range(5), repeat=self.num_agents):
                    Oj = T_mapf(self.label, self.goal, self.layout, Oi, joint_a)
                    A = joint_a[self.label]
                    j = Omega.index(Oj)
                    reward = R_mapf(self.label, self.goal, Oi, Oj)

                    # Only cares about the landing state
                    if Oj == 'EDGECONFLICT':
                        R[idx, A, i, j] = reward
                        continue

                    coef = 1
                    for op_id, pi_id in enumerate(joint_idxs):
                        if op_id == self.label:
                            continue
                        coef *= beliefs_pi[op_id][pi_id][hash(Oi[op_id])][joint_a[op_id]]
                    R[idx, A, i, j] += coef * reward

        Q_mdp = np.zeros(shape=(num_all_joint_pi, num_all_obs, 5))
        for idx in tqdm(range(num_all_joint_pi)):
            VI = mdp.ValueIteration(T[idx], R[idx], discount=0.9)
            VI.run()
            V_opt = VI.V
            Q = np.zeros(shape=(num_all_obs, 5))
            for a in range(5):
                Q[:, a] = 0.9 * T[idx, a] @ V_opt + np.sum(T[idx, a] * R[idx, a], axis=1)
            Q_mdp[idx] = Q

        return Q_mdp
