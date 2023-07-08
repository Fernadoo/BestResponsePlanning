from search_agent import dijkstra
from utils import move, hash, soft_max, enumerate_all

import os
from collections import namedtuple
from itertools import product
from copy import deepcopy

import numpy as np
from mdptoolbox import mdp


"""
MAPF transition & reward for sing-agent MDP formulation
"""


def T_mapf(label, goal, layout, locations, action_profile):
    """
    The multi-agent env transition:
    given a tuple of locations and an action profile,
    returns the successor locations.
    """
    # If edge conflict, no valid transition
    if locations == 'EDGECONFLICT':
        return 'EDGECONFLICT'

    # If goal, no valid transition
    if locations[label] == goal:
        return tuple(locations)

    # If vertex conflict, no valid transition
    for i, other_loc in enumerate(locations):
        if i != label and other_loc == locations[label]:
            return tuple(locations)

    nrows = len(layout)
    ncols = len(layout[0])
    succ_locations = []
    for i in range(len(locations)):
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
        if i != label:
            if other_loc == locations[label] and\
                    succ_locations[label] == locations[i]:
                return 'EDGECONFLICT'

    return tuple(succ_locations)


def R_mapf(label, goal, pred_locs, succ_locs):
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
        if i != label and other_loc == succ_locs[label]:
            return -1000

    if succ_locs[label] == goal:
        return 1000
    return -1


"""
MDP agent:
[Assuming the belief will NOT change in the futher]
1. Initialize the belief about others
2. For each iteration:
    i) Construct the MDP induced by the belief
    ii) Do the next one step as the opt policy indicates
3. Observe how the others played and update the belief
"""


class MDPAgent(object):
    """docstring for MDPAgent"""

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
                reward = R_mapf(self.label, self.goal, Si, Sj)

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


"""
POMDP agent:
[Also planning for future belief update]
1. Initialize the belief about others
2. For each iteration:
    i) Construct the POMDP induced by all others' possible policies
    ii) Do the next one step as the opt policy indicates
3. Observe how the others played and update the belief
"""


class POMDPAgent(MDPAgent):
    """docstring for POMDPAgent"""

    def __init__(self, label, goal, belief_update=False):
        super(POMDPAgent, self).__init__(label, goal, belief_update)

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

        self.print_belief(self.beliefs)
        Si = self.S.index(locations)
        action = self.policy[Si]
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
        exit()
        return policy

    def write_pomdp(self, T, Obs, R, discount=0.95):
        # Write a pomdp file
        pomdp_file_name = f'pomdp-solve/problems/{self.label}.POMDP'
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
        # Solve the pomdp
        solver = './pomdp-solve/pomdp-solve-os-x.bin'
        sol_file_prefix = f'{pomdp_file_name[:-6]}_sol'
        os.system(f'{solver} -horizon {h}'
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
