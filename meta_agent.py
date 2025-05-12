import numpy as np

from mdp_agent import MDPAgent


class MetaAgent(MDPAgent):
    """docstring for MetaAgent"""

    def __init__(self, label, goal, meta_policy, belief_update=True, soft_update=1e-4, verbose=False):
        super(MetaAgent, self).__init__(label, goal, belief_update, soft_update, verbose)
        self.policy = meta_policy

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        # if self.policy is None:
        #     self.policy = PPO.load('pretrained/MetaPPO_small_8e6.zip')
        #     print('a')

        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(self.beliefs, self.prev_locations, prev_actions)

        if self.verbose:
            self.print_belief(self.beliefs)

        self.prev_locations = locations

        goal = self.goal
        loc = [locations[self.label]] + list(locations)[:self.label] + list(locations)[self.label + 1:]
        belief_probs = self.beliefs[1]
        obs = np.concatenate([goal / np.array(self.layout.shape),
                              np.array(loc).reshape(-1) / np.array(N * self.layout.shape),
                              np.concatenate(belief_probs)])
        # print(obs)
        best_action = self.policy.predict(obs, deterministic=True)[0]

        return best_action
