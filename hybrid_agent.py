import numpy as np

from cbs_agent import CBSEval
from utils import get_avai_actions_mapf, man_dist
from search_agent import astar
from mcts_agent import AsymmetricTreeSearch


class HybridAgent(AsymmetricTreeSearch):
    """
    HybridAgent combining safe agents and mcts
    """

    def act(self, state):
        N, prev_actions, locations, layout = state

        if self.plan is None:
            self.plan = astar(locations[self.label], self.goal, layout)
        if locations[self.label] == self.goal:
            return 0

        action = self.plan[self.round]
        self.round += 1

        
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
