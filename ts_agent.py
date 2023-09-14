from mdp_agent import MDPAgent
from utils import (T_mapf, R_mapf, get_avai_actions_mapf,
                   hash, man_dist, enumerate_all)

from copy import deepcopy
from itertools import product
from queue import Queue

from tqdm import tqdm
from mdptoolbox import mdp
import numpy as np


class TreeNode(object):
    """
    Node structure used in tree search:
    Parameters:
        type: 'MAX' or 'EXP'
        height: >= 0
        locations: location tuple
        beliefs: only MAX nodes are with valid beliefs
        val: backpropagated long run return
        reward: immediate reward from the parent node and the branch action
    """

    def __init__(self, tp, h, locations, reward, beliefs=None, prev_actions=None, history=[]):
        if tp not in ['MAX', 'EXP']:
            raise ValueError('No such node type!')
        self.type = tp
        self.height = h
        self.locations = locations
        self.val = None
        self.reward = reward
        self.children = []
        self.beliefs = beliefs
        self.prev_actions = prev_actions
        self.history = history


class UniformTreeSearchAgent(MDPAgent):
    """docstring for UniformTreeSearchAgent"""

    def __init__(self, label, goal,
                 belief_update=True, depth=2, node_eval='MDP', discount=0.9,
                 check_repeated_states=False):
        super(UniformTreeSearchAgent, self).__init__(label, goal, belief_update)

        # limited depth for expansion, -1 means to expand until termination
        self.depth = depth

        # evaluation mode: 'IMMED', 'MDP', 'HEU'
        self.node_eval = node_eval
        self.discount = discount

        # whether different eval for repeated states
        self.check_repeated_states = check_repeated_states

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # Formulate a search tree based on the current state and belief
        # Construc the search at the beginning
        if self.policy is None:
            self.policy = self.tree_search(locations)

        # Update the belief and re-construct the search tree
        if prev_actions is not None:
            if self.belief_update:
                self.beliefs = self.update_belief(self.beliefs,
                                                  self.prev_locations,
                                                  prev_actions,
                                                  soft=1e-2)
            self.policy = self.tree_search(locations, prev_actions)

        self.print_belief(self.beliefs)

        # Enquire the policy
        action_values = list(map(lambda c: c.val, self.policy.children))
        print(action_values)
        action = np.argmax(action_values)
        if action not in get_avai_actions_mapf(locations[self.label], self.layout):
            action = 0
        print(action)
        self.prev_locations = locations
        return action

    def tree_search(self, curr_locs, prev_actions=None):
        if getattr(self, 'root', None) is None:
            self.root = TreeNode('MAX', 0, curr_locs, 0, self.beliefs)
            self.self_br = tuple(range(5))
            self.oppo_br = tuple(product(range(5), repeat=self.num_agents - 1))
        else:
            # Reuse the previous search tree
            self_a_idx = prev_actions[self.label]
            del prev_actions[self.label]
            oppo_a = tuple(prev_actions)
            oppo_a_idx = tuple(product(range(5), repeat=self.num_agents - 1)).index(oppo_a)
            self.root = self.root.children[self_a_idx].children[oppo_a_idx]

        node_to_eval = self.expand(self.root)
        for _ in tqdm(range(len(self.expanded.queue) + 1)):
            if node_to_eval is None:
                break
            self.evaluate(node_to_eval)
            node_to_eval = self.expand(self.root)
        self.backup(self.root)
        self.expanded = None
        return self.root

    def expand(self, selected_node):
        # Grow the complete tree until the predefined depth
        # By layer-first-search
        if getattr(self, 'expanded', None) is not None:
            if self.expanded.empty():
                return None
            next_node = self.expanded.get()
            return next_node

        max_height = 2 * self.depth + self.root.height
        q = Queue()
        q.put(selected_node)
        while not q.empty():
            curr_node = q.get()
            if curr_node.height >= max_height:
                self.expanded = q
                return curr_node

            # Possibility 1: already expanded
            if len(curr_node.children) != 0:
                successors = curr_node.children
                for succ_node in successors:
                    q.put(succ_node)
                continue

            # Possibility 2: newly expanded
            successors = []
            if curr_node.type == 'MAX':
                fanout = self.self_br
                for a in fanout:
                    succ_node = TreeNode('EXP', curr_node.height + 1,
                                         locations=curr_node.locations,
                                         reward=curr_node.reward,
                                         beliefs=curr_node.beliefs,
                                         prev_actions=a,
                                         history=curr_node.history)
                    successors.append(succ_node)
                    q.put(succ_node)

            elif curr_node.type == 'EXP':
                fanout = self.oppo_br
                for other_a in fanout:
                    # MAX nodes are associated with valid belief
                    prev_joint_a = list(other_a)
                    prev_joint_a.insert(self.label, curr_node.prev_actions)
                    if self.belief_update and curr_node.locations != 'EDGECONFLICT':
                        succ_beliefs = self.update_belief(curr_node.beliefs,
                                                          curr_node.locations,
                                                          prev_joint_a,
                                                          soft=1e-2)
                    else:
                        succ_beliefs = curr_node.beliefs
                    succ_locs = T_mapf(self.label, self.goal, self.layout,
                                       curr_node.locations, prev_joint_a)
                    rwd = R_mapf(self.label, self.goal, curr_node.locations, succ_locs, penalty=1e3)
                    hist = deepcopy(curr_node.history)
                    hist.append(curr_node.locations)
                    succ_node = TreeNode('MAX', curr_node.height + 1,
                                         locations=succ_locs,
                                         reward=rwd,
                                         beliefs=succ_beliefs,
                                         prev_actions=prev_joint_a,
                                         history=hist)
                    successors.append(succ_node)
                    q.put(succ_node)

            curr_node.children = successors

        raise RuntimeError('The expansion process went wrong!')

    def evaluate(self, node_to_eval):
        """
        Return immediate reward or,
        Construct an fix-belief MDP for node evaluation
        """
        if getattr(self, 'beliefs', None) is None or\
                getattr(self, 'layout', None) is None:
            raise RuntimeError("Get invalid beliefs, "
                               "or invalid layout!")

        beliefs_pi, beliefs_prob = node_to_eval.beliefs
        beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
        beliefs_num[self.label] = range(1)

        self.beliefs_num = beliefs_num  # for later reuse

        if self.check_repeated_states:
            if node_to_eval.locations in node_to_eval.history:
                first_occur = node_to_eval.history.index(node_to_eval.locations)
                node_to_eval.val = node_to_eval.reward - 10 * (len(node_to_eval.history) - first_occur)
                return

        if self.node_eval == 'IMMED':
            node_to_eval.val = node_to_eval.reward
            return
        if self.node_eval == 'HEU':
            if node_to_eval.locations == 'EDGECONFLICT':
                node_to_eval.val = -1000
                return
            loc = node_to_eval.locations[self.label]
            node_to_eval.val = 1000 - man_dist(loc, self.goal)
            return

        # Get all possible states
        if getattr(self, 'S', None) is None:
            self.S = enumerate_all(self.num_agents, self.layout)
            self.S.append('EDGECONFLICT')
            self.num_all = len(self.S)

        S = self.S
        num_all = self.num_all

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
        VI = mdp.ValueIteration(T, R, discount=self.discount)
        VI.run()
        node_to_eval.val = VI.V[S.index(node_to_eval.locations)]

    def backup(self, node):
        if node.height == 2 * self.depth + self.root.height:
            return node.val
        if node.type == 'MAX':
            child_values = list(map(lambda n: self.backup(n), node.children))
            # print(child_values)
            node.val = max(child_values)
            return node.val
        else:
            child_values = list(map(lambda n: self.backup(n), node.children))
            # print(child_values)
            probs = np.zeros(len(node.children))
            rewards = np.zeros(len(node.children))
            for idx, child in enumerate(node.children):
                joint_a = child.prev_actions
                Si = node.locations
                if Si == 'EDGECONFLICT':
                    probs[idx] = 1
                    continue

                beliefs_pi, beliefs_prob = node.beliefs
                for joint_idxs in product(*self.beliefs_num):
                    sub_prob = 1
                    for op_id, pi_id in enumerate(joint_idxs):
                        if op_id == self.label:
                            continue
                        sub_prob *= (
                            beliefs_pi[op_id][pi_id][hash(Si[op_id])][joint_a[op_id]]
                            * beliefs_prob[op_id][pi_id]
                        )
                    probs[idx] += sub_prob

                rewards[idx] = child.reward

            probs = probs / np.sum(probs)
            node.val = np.dot(np.array(child_values) * self.discount + rewards, probs)
            return node.val
