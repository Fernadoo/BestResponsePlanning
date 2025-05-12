from cbs_agent import CBSEval
from mdp_agent import MDPAgent
from uts_agent import TreeNode
from utils import (T_mapf, R_mapf, get_avai_actions_mapf,
                   hash, man_dist)

from copy import deepcopy
from itertools import product

from tqdm import tqdm
import numpy as np


class AsymmetricTreeSearch(MDPAgent):
    """
    MCTS with both chance nodes and decion nodes
    Implement asymmetric tree growth
    """

    def __init__(self, label, goal,
                 belief_update=True, soft_update=1e-6,
                 verbose=False, discount=0.95,
                 max_it=100,
                 explore_c=1,
                 pUCB=False,
                 pb_c=(1.25, 19652.0),
                 node_eval='HEU',
                 sample_eval=0,
                 sample_select=10,
                 reward_scheme=None):
        super(AsymmetricTreeSearch, self).__init__(label, goal, belief_update, soft_update, verbose)

        self.discount = discount
        self.max_it = int(max_it)
        self.c = explore_c  # for value guided mcts
        self.pUCB = pUCB  # for policy-value guided mcts
        self.pb_c_init, self.pb_c_base = pb_c

        # evaluation mode: 'IMMED', 'MDP', 'HEU', 'NN', 'CBS'
        self.node_eval = node_eval

        self.sample_eval = sample_eval  # for cbs eval
        self.sample_select = sample_select  # for node selection

        if reward_scheme:
            self.penalty = reward_scheme['collision']
            self.goal_reward = reward_scheme['goal']
        else:
            self.penalty = 200  # 2e3
            self.goal_reward = 50  # 1e3

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N

            if self.node_eval == 'CBS' and\
                    getattr(self, 'cbs_estimator', None) is None:
                self.cbs_estimator = CBSEval(
                    self.label, self.goal, self.layout, self.goal_reward, self.penalty, self.discount
                )

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
                                                  soft=self.soft_update)
            self.policy = self.tree_search(locations, prev_actions)

        if self.verbose:
            self.print_belief(self.beliefs)
            print(f"MANH dist to goal: {man_dist(locations[self.label], self.goal)}")

        avai_actions = get_avai_actions_mapf(locations[self.label], self.layout)
        unavai_actions = [a for a in range(5) if a not in avai_actions]
        child_num_visits = list(map(lambda c: c.num_visit, self.policy.children))
        child_num_visits = np.array(child_num_visits, dtype=float)
        action_values = list(map(lambda c: c.val / c.num_visit, self.policy.children))
        action_values = np.array(action_values, dtype=float)
        if self.verbose:
            print(child_num_visits, action_values)

        if self.pUCB:
            child_num_visits[[unavai_actions]] = -np.inf
            action = np.argmax(child_num_visits)
        else:
            action_values[[unavai_actions]] = -np.inf
            action = np.argmax(action_values)

        # action_values = list(map(lambda c: c.val / c.num_visit, self.policy.children))
        # print(action_values)
        # action_values = np.array(action_values, dtype=float)
        # action_values[[unavai_actions]] = -np.inf
        # action = np.argmax(action_values)

        # if action not in get_avai_actions_mapf(locations[self.label], self.layout):
        #     action = 0

        if getattr(self, 'prev_locations', None)\
                and locations == self.prev_locations\
                and action == 0:
            self.max_it = max(7, int(self.max_it * 0.6))

        self.prev_locations = locations
        return action

    def tree_search(self, curr_locs, prev_actions=None):
        # if getattr(self, 'root', None) is None:
        #     self.root = TreeNode('MAX', 0, curr_locs, 0, self.beliefs)

        #     # self.self_br = tuple(range(5))
        #     # self.oppo_br = tuple(product(range(5), repeat=self.num_agents - 1)) #TODO

        # else:
        #     # Reuse the previous search tree
        #     self_a_idx = prev_actions[self.label]
        #     oppo_a = tuple(prev_actions[:self.label] + prev_actions[self.label + 1:])
        #     # oppo_a_idx = tuple(product(range(5), repeat=self.num_agents - 1)).index(oppo_a) #TODO
        #     # succ_root = self.root.children[self_a_idx].children[oppo_a_idx]
        #     if oppo_a in self.root.children[self_a_idx].child_action_indicies:
        #         oppo_a_idx = self.root.children[self_a_idx].child_action_indicies.index(oppo_a)
        #         succ_root = self.root.children[self_a_idx].children[oppo_a_idx]

        #     # However, chances are that oppo_a have not been sampled previously
        #     else:
        #     # if succ_root == 'NULL':
        #         rwd = R_mapf(self.label, self.goal, self.prev_locations, curr_locs)
        #         succ_root = TreeNode('MAX', self.root.height + 2,  # exp+1 and max+1
        #                              locations=curr_locs,
        #                              reward=rwd,
        #                              beliefs=self.beliefs,
        #                              prev_actions=prev_actions)

        #         # self.root.children[self_a_idx].children[oppo_a_idx] = succ_root
        #         self.root.children[self_a_idx].children.append(succ_root)
        #         self.root.children[self_a_idx].child_action_indicies.append(oppo_a)
        #         succ_root.set_parent(self.root.children[self_a_idx])

        #     self.root = succ_root

        self.root = TreeNode('MAX', 0, curr_locs, 0, self.beliefs)
        max_height = 0
        if self.verbose:
            iterator = tqdm(range(self.max_it))
        else:
            iterator = range(self.max_it)
        for it in iterator:
            node_to_exp = self.select(self.root, iteration=it)
            node_to_eval = self.expand(node_to_exp)
            max_height = max(max_height, node_to_eval.height)
            info = self.evaluate(node_to_eval)
            self.backup(node_to_eval, info)
        if self.verbose:
            print(f'Took {self.max_it} simus, lookahead for {max_height / 2} steps')
        return self.root

    def select(self, root, iteration):
        """
        Best-first node selection:
        V_s / N_s + sqrt(2N / N_s)
        Note that Q(s, a) = V_s / N_s is not stationary
        Upon each chosen action, sample a succ_state by the transition model
        """
        curr_node = root  # the iterating node is always a MAX node
        selected_a = 0
        while True:
            # a state whose action has not been fully expanded or is a leaf node
            if len(curr_node.children) < 5:
                return curr_node

            # or, iteratively find the best action with UCB heuristic
            if self.pUCB:
                Qs = _puct(curr_node)
            else:
                Qs = _vanilla_uct(curr_node, iteration)

            if curr_node.locations == 'EDGECONFLICT':
                selected_a = 0
            else:
                avai_actions = get_avai_actions_mapf(curr_node.locations[self.label], self.layout)
                unavai_actions = [a for a in range(5) if a not in avai_actions]
                Qs[[unavai_actions]] = -np.inf
                selected_a = np.argmax(Qs)

            oppo_a_idx = self.sample_from_belief(curr_node, selected_a)
            curr_node = curr_node.children[selected_a].children[oppo_a_idx]

    def expand(self, node_to_exp):
        """
        Given a MAX node, try a not-chosen action, and a new EXP node
        Then sample a new child MAX node with belief revision
        """
        # correct?: check if node_to_exp is new
        if node_to_exp.num_visit == 0:
            return node_to_exp

        expand_a = len(node_to_exp.children)
        succ_node = TreeNode('EXP', node_to_exp.height + 1,
                             locations=node_to_exp.locations,
                             beliefs=node_to_exp.beliefs,
                             prev_actions=expand_a)
        node_to_exp.children.append(succ_node)
        succ_node.set_parent(node_to_exp)
        # succ_node.children = ['NULL' for i in range(5 ** (self.num_agents - 1))] #TODO
        oppo_a_idx = self.sample_from_belief(node_to_exp, expand_a)

        return node_to_exp.children[expand_a].children[oppo_a_idx]

    def sample_from_belief(self, node, action, epsilon=1e-2):
        """
        Given a MAX node and an action,
        sample an action profile for the other agents,
        return the index of the sampled profile
        """
        np.random.seed(618)
        oppo_a = []
        beliefs_pi, beliefs_prob = node.beliefs
        for i in range(self.num_agents):
            if i == self.label:
                continue
            if node.locations == 'EDGECONFLICT':
                oppo_a.append(0)  # any action is invalid, stop by default
                continue

            action_dist = np.zeros(5)
            for _ in range(self.sample_select):
                pi = np.random.choice(beliefs_pi[i], p=beliefs_prob[i])
                action_dist += np.add(pi[hash(node.locations[i])], epsilon)
            a_i = np.random.choice(range(5), p=action_dist / np.sum(action_dist))
            oppo_a.append(a_i)
        oppo_a_tuple = tuple(oppo_a)
        # oppo_a_idx = tuple(product(range(5), repeat=self.num_agents - 1)).index(oppo_a_tuple) #TODO

        if oppo_a_tuple in node.children[action].child_action_indicies:
            return node.children[action].child_action_indicies.index(oppo_a_tuple)

        else:
        # if node.children[action].children[oppo_a_idx] == 'NULL':
            prev_joint_a = deepcopy(oppo_a)
            prev_joint_a.insert(self.label, action)
            if self.belief_update and node.locations != 'EDGECONFLICT':
                succ_beliefs = self.update_belief(node.beliefs,
                                                  node.locations,
                                                  prev_joint_a,
                                                  soft=self.soft_update)
            else:
                succ_beliefs = node.beliefs
            succ_locs = T_mapf(self.label, self.goal, self.layout,
                               node.locations, prev_joint_a)
            rwd = R_mapf(self.label, self.goal, node.locations, succ_locs,
                         penalty=self.penalty, goal_reward=self.goal_reward)
            succ_node = TreeNode('MAX', node.height + 2,  # exp+1 and max+1
                                 locations=succ_locs,
                                 reward=rwd,
                                 beliefs=succ_beliefs,
                                 prev_actions=prev_joint_a)
            # node.children[action].children[oppo_a_idx] = succ_node

            node.children[action].children.append(succ_node)
            node.children[action].child_action_indicies.append(oppo_a_tuple)

            succ_node.set_parent(node.children[action])
            return -1

        # return oppo_a_idx


    def evaluate(self, node_to_eval):
        """
        By enquiring contextual MDP: M(belief)
        """
        if self.node_eval.startswith('HEU'):
            val = 0
            if node_to_eval.locations == 'EDGECONFLICT':
                return val
            loc = node_to_eval.locations[self.label]
            val += 1000 - man_dist(loc, self.goal)
            if self.node_eval.endswith('-C'):
                for j, loc_j in enumerate(node_to_eval.locations):
                    val += 1 / 5 * man_dist(loc_j, loc)
            return val

        if self.node_eval == 'CBS':
            if node_to_eval.locations == 'EDGECONFLICT':
                # TODO: Assign policy prior or not?
                # return 0
                node_to_eval.policy_prior = np.ones(5) / 5
                return -self.penalty

            policy_prior, value = self.cbs_estimator.eval(
                node_to_eval.locations, node_to_eval.beliefs, sample_eval=self.sample_eval
            )
            # print(value)
            if self.pUCB:
                node_to_eval.policy_prior = policy_prior
            return value

        else:
            return 0

    def backup(self, evaled_node, info):
        future_val = info
        curr_node = evaled_node

        # future_val = self.discount * future_val + curr_node.reward
        curr_node.val += future_val
        curr_node.num_visit += 1
        while getattr(curr_node, 'parent', None) is not None:
            # MAX -> EXP
            future_val = self.discount * future_val + curr_node.reward
            curr_node = curr_node.parent
            curr_node.val += future_val
            curr_node.num_visit += 1

            # EXP -> MAX
            curr_node = curr_node.parent
            # future_val = self.discount * future_val + curr_node.reward
            curr_node.val += future_val
            curr_node.num_visit += 1


def _vanilla_uct(node, it,
                 *,
                 c=2):
    c = c * np.exp(-0.05 * it)
    c = 1
    node_visit = node.num_visit
    child_visit = np.array(list(map(lambda n: n.num_visit, node.children)))
    Qs = _qtransform(node)
    ucb = c * np.sqrt(2 * np.log(node_visit) / child_visit)
    tie_breaking_noise = np.random.uniform(size=len(child_visit)) * 1e-10
    return Qs + ucb + tie_breaking_noise


def _puct(node,
          *,
          pb_c_init=0.5,  # deepmind: 1.25,
          pb_c_base=19652.0,
          seed=0):
    """
    The pUCT formula from muZero:
    given a `parent` node, return the pUCT score for its children.
    `1 + child_visit` because it is initialzed to 0
    """
    # TODO
    # 1. num_visit init to 0 or 1?
    # 2. how q be normalized to [0, 1]?
    #    2.1 should be pass a parent node?
    node_visit = node.num_visit
    child_visit = np.array(list(map(lambda n: n.num_visit, node.children)))
    # TODO: if a leaf node is reused, num_visit will equal to child_visit + 1
    # if node_visit != sum(child_visit):
    #     raise ValueError(f"{node_visit} not equal to sum of {child_visit}")
    prior_probs = (node.policy_prior + 1e-1) / np.sum(node.policy_prior + 1e-1)
    pb_c = pb_c_init + np.log((node_visit + pb_c_base + 1) / pb_c_base)
    policy_prior = prior_probs * np.sqrt(node_visit) * pb_c / (child_visit)
    Qs = _qtransform(node)
    tie_breaking_noise = np.random.uniform(size=len(child_visit)) * 1e-10
    return Qs + policy_prior + tie_breaking_noise


def _qtransform(node, *, epsilon=1e-8):
    """
    Given a parent node,
    return Qs for its children normalized to [0, 1].
    Normalization is done w.r.t. the parent and siblings.
    """
    child_Qs = np.array(list(map(lambda x: x.val / x.num_visit, node.children)))
    node_Q = node.val / node.num_visit
    lo = np.minimum(node_Q, np.min(child_Qs))
    hi = np.maximum(node_Q, np.max(child_Qs))
    return (child_Qs - lo) / np.maximum((hi - lo), epsilon)
