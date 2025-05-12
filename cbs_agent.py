import os
import random
import re
import time
from copy import deepcopy
from itertools import product

import numpy as np
from tqdm import tqdm

from mdp_agent import MDPAgent

ALGOLIB = {
    "eecbs": "cbs-solve/eecbs",
}
TMP_PREFIX = '../../../../../Desktop/cbstmp'  # change to a preferred one


class CBSAgent(MDPAgent):
    def __init__(self, label, goal, belief_update=True, soft_update=1e-6, verbose=False,
                 goal_reward=1e3, penalty=3e4, discount=0.9, sample_eval=10):
        super(CBSAgent, self).__init__(label, goal, belief_update, soft_update, verbose)
        self.goal_reward = goal_reward
        self.penalty = penalty
        self.discount = discount
        self.sample_eval = sample_eval

    def act(self, state):
        N, prev_actions, locations, layout = state
        if self.beliefs is None:
            self.beliefs = self.init_belief(N, layout)
            self.layout = layout
            self.num_agents = N
        if locations[self.label] == self.goal:
            return 0

        if getattr(self, 'cbs_estimator', None) is None:
            self.cbs_estimator = CBSEval(
                self.label, self.goal, self.layout, self.goal_reward, self.penalty, self.discount
            )

        if prev_actions is not None and self.belief_update:
            self.beliefs = self.update_belief(self.beliefs, self.prev_locations, prev_actions,
                                              soft=self.soft_update)

        if self.verbose:
            self.print_belief(self.beliefs)

        # t1 = time.time()
        policy_prior, value = self.cbs_estimator.eval(
            locations, self.beliefs, sample_eval=self.sample_eval
        )
        action = np.argmax(policy_prior)
        # t2 = time.time()
        # print(t2 - t1)

        self.prev_locations = locations
        return action

    def close(self):
        if getattr(self, 'cbs_estimator', None):
            self.cbs_estimator.close()


class CBSEval():
    """docstring for planner-based estimator for policy priors and values"""

    def __init__(self, label, goal, layout, goal_reward, penalty, discount):
        self.label = label
        self.goal = goal
        self.layout = layout
        self.rewards = {'goal': goal_reward, 'ow': -1, 'collision': -penalty}
        self.discount = discount

        empty_cells = np.array(np.where(self.layout == 0)).T
        empty_cells = list(map(tuple, empty_cells))
        del empty_cells[empty_cells.index(self.goal)]
        self.type2goal = empty_cells

        self.hash_id = hash(time.time())
        # print(self.hash_id)
        self.layout_file = write_layout_file(self.layout, self.hash_id)

    def close(self):
        os.system(f"rm -f {self.layout_file} {self.agent_file} {self.sol_file} >> {TMP_PREFIX}/tmp.log")

    def eval(self, locations, beliefs, sample_eval=0):
        """
        Cannot start with vertex conflict and overlapped goals
        e.g.,

        version 1
        0   tmp_map tmp_h   tmp_w   3   6   3   6   tmp_opt
        0   tmp_map tmp_h   tmp_w   3   6   1   6   tmp_opt

        will generates

        Agent 0: (6,2)->
        Agent 1: (5,3)->(5,2)->(5,1)->(6,1)->
        """
        other_locs = locations[:self.label] + locations[self.label + 1:]
        if locations[self.label] in other_locs:
            # return np.array([1, 0, 0, 0, 0]), self.rewards['collision']
            return np.ones(5) / 5, self.rewards['collision']

        # Need to check collision first, in case collide at goal the last step
        # print(locations[self.label], self.goal)
        if locations[self.label] == self.goal:
            return np.array([1, 0, 0, 0, 0]), self.rewards['goal']

        # rule out overlapping agents
        valid_agent_ids = [self.label]
        valid_locations = [locations[self.label]]
        for i in range(len(locations)):
            if locations[i] in valid_locations:
                continue
            valid_agent_ids.append(i)
            valid_locations.append(locations[i])

        num_agents = len(valid_agent_ids)
        beliefs_pi, beliefs_prob = beliefs

        if sample_eval:
            np.random.seed(618)
            acc_policy_priors = np.zeros(5)
            acc_values = 0
            valid_samples = 0
            actual_samples = 0
            while valid_samples < sample_eval:
                # TODO: May cause actual_samples >> valid_samples when agents are too dense
                # Potential solution: importance sampling?
                actual_samples += 1

                valid_goals = [self.goal]
                INVALID_GOAL = False
                for i in valid_agent_ids:
                    if i == self.label:
                        continue
                    else:
                        t_i = np.random.choice(len(beliefs_prob[i]), p=beliefs_prob[i])
                        g = self.type2goal[t_i]
                        if g in valid_goals:
                            INVALID_GOAL = True
                            break
                        valid_goals.append(g)
                if INVALID_GOAL:
                    continue

                # valid_goals = [self.goal]
                # for i in valid_agent_ids:
                #     if i == self.label:
                #         continue
                #     else:
                #         while True:
                #             t_i = np.random.choice(len(beliefs_prob[i]), p=beliefs_prob[i])
                #             g = self.type2goal[t_i]
                #             if g not in valid_goals:
                #                 break
                #         valid_goals.append(g)

                valid_samples += 1

                # If the opponent is already at her goal, treat it as an obstacle
                revised_valid_locations = []
                revised_valid_goals = []
                revised_layout = deepcopy(self.layout)
                REVISED = False
                for i, loc in enumerate(valid_locations):
                    if loc == valid_goals[i]:
                        revised_layout[loc] = 1
                        REVISED = True
                        continue
                    revised_valid_locations.append(loc)
                    revised_valid_goals.append(valid_goals[i])
                if REVISED:
                    # print(locations)
                    # print(valid_locations, valid_goals)
                    # print(revised_valid_locations, revised_valid_goals)
                    # print(revised_layout)
                    self.layout_file = write_layout_file(revised_layout, self.hash_id)
                num_agents = len(revised_valid_locations)

                self.agent_file = write_agent_file(revised_valid_locations, revised_valid_goals, self.hash_id)
                # print(self.agent_file)
                # self.agent_file = write_agent_file(valid_locations, valid_goals, self.hash_id)
                self.sol_file = call_solver(self.layout_file, self.agent_file, num_agents, self.hash_id)

                # restore the layout file
                if REVISED:
                    self.layout_file = write_layout_file(self.layout, self.hash_id)

                if self.sol_file is None:
                    val_stop_forever = self.rewards['ow'] / (1 - self.discount)
                    acc_policy_priors += np.array([1, 0, 0, 0, 0])
                    acc_values += val_stop_forever
                    continue

                # path = read_sol(self.sol_file, agent_id=self.label)
                path = read_sol(self.sol_file, agent_id=0)

                curr_loc = np.array(locations[self.label], dtype=int)
                next_loc = np.array(path[1], dtype=int)
                next_action = dxdy2action(tuple(next_loc - curr_loc))
                # print(curr_loc, next_loc, next_action)
                assert next_action is not None, f"invalid CBS call: {(valid_locations, valid_goals)}"
                acc_policy_priors += np.eye(5)[next_action]

                n = len(path) - 1
                # print(n)
                acc_values +=\
                    self.rewards['ow'] *\
                    self.discount * (1 - self.discount ** (n - 1)) / (1 - self.discount) +\
                    self.rewards['goal'] *\
                    self.discount ** n

            # print(actual_samples)
            return acc_policy_priors / sample_eval, acc_values / sample_eval

        else:
            valid_beliefs_num = [[None]] +\
                [range(len(Pi_i)) for i, Pi_i in enumerate(beliefs_pi)
                 if i in valid_agent_ids and i != self.label]

            # beliefs_num = list(map(lambda Pi_i: range(len(Pi_i)), beliefs_pi))
            # beliefs_num[self.label] = range(1)
            self.valid_joint_pi_indices = list(product(*valid_beliefs_num))

            probs = np.ones(len(self.valid_joint_pi_indices))
            policy_priors = np.zeros((len(self.valid_joint_pi_indices), 5))
            values = np.zeros(len(self.valid_joint_pi_indices))
            for idx, joint_pi_idx in enumerate(self.valid_joint_pi_indices):
                valid_goals = [self.goal]
                INVALID_GOAL = False
                for i, t_i in enumerate(joint_pi_idx):
                    if i == 0:  # modelling agent in the first slot
                        continue
                    else:
                        g = self.type2goal[t_i]
                        if g in valid_goals:
                            INVALID_GOAL = True
                            break
                        valid_goals.append(g)
                        probs[idx] *= beliefs_prob[valid_agent_ids[i]][t_i]
                if INVALID_GOAL:
                    continue

                self.agent_file = write_agent_file(valid_locations, valid_goals, self.hash_id)
                self.sol_file = call_solver(self.layout_file, self.agent_file, num_agents, self.hash_id)
                path = read_sol(self.sol_file, agent_id=0)

                curr_loc = np.array(locations[self.label], dtype=int)
                next_loc = np.array(path[1], dtype=int)
                next_action = dxdy2action(tuple(next_loc - curr_loc))
                policy_priors[idx][next_action] = 1

                n = len(path) - 1
                values[idx] =\
                    self.rewards['ow'] *\
                    self.discount * (1 - self.discount ** (n - 1)) / (1 - self.discount) +\
                    self.rewards['goal'] *\
                    self.discount ** n

            assert np.isclose(np.sum(probs), 1), (probs, probs[1:].sum(), joint_pi_idx)

        return probs @ policy_priors, probs @ values


def write_layout_file(layout, hash_id, layout_file_prefix=TMP_PREFIX):
    def marker2char(marker):
        if marker:
            return '@'
        else:
            return '.'

    layout_file = f"{layout_file_prefix}/{hash_id}.map"
    with open(layout_file, 'w') as lf:
        lf.write(f"type warehouse\n")
        lf.write(f"height {layout.shape[0]}\n")
        lf.write(f"width {layout.shape[1]}\n")
        lf.write(f"map\n")
        layoutlines = []
        for i in range(layout.shape[0]):
            row = ''.join(list(map(marker2char, layout[i]))) + '\n'
            layoutlines.append(row)
        lf.writelines(layoutlines)

    return layout_file


def write_agent_file(locations, goals, hash_id, agent_file_prefix=TMP_PREFIX):
    num_agents = len(locations)
    agent_file = f"{agent_file_prefix}/{hash_id}.scen"
    with open(agent_file, 'w') as af:
        af.write(f"version 1\n")
        alines = []
        for i in range(num_agents):
            x, y = locations[i]
            g_x, g_y = goals[i]
            line = f"0\ttmp_map\ttmp_h\ttmp_w\t{y}\t{x}\t{g_y}\t{g_x}\ttmp_opt\n"
            alines.append(line)
        af.writelines(alines)

    return agent_file


def call_solver(layout_file, agent_file, num_agents, hash_id,
                alg='eecbs',
                sol_file_prefix=TMP_PREFIX,
                timeout=10,
                subopt=1.2):
    solver_path = ALGOLIB[alg]
    sol_file = f"{sol_file_prefix}/{hash_id}paths.txt"
    if alg == 'eecbs':
        args = [
            f"{solver_path}",
            f"-m {layout_file}",
            f"-a {agent_file}",
            f"-k {num_agents}",
            f"--outputPaths={sol_file}",
            f"-t {timeout}",
            f"--suboptimality={subopt}",
        ]
        # TODO: a more elegant way via os.subprocess
        cmd = " ".join(args)
        # print(cmd)
        os.system(f"{cmd} > {TMP_PREFIX}/{hash_id}tmp.log 2>&1")
        # print(args)
        # subprocess.run(args)
        if re.split(",| ", open(f"{TMP_PREFIX}/{hash_id}tmp.log", 'r').read().split(": ")[1])[0] != 'Succeed':
            # 1. There is a valid solution, but failed to find it in time;
            # 2. There is no valid solution, as agents at goals wont move.
            return None

    return sol_file


def read_sol(sol_file, agent_id):
    with open(sol_file, 'r') as sf:
        line = None
        for i in range(agent_id + 1):
            line = sf.readline()  # e.g., Agent 0: (16,5)->(17,5)->(17,6)->
        chunks = re.split(': |->', line)
        found_id = eval(chunks[0].split()[-1])
        assert found_id == agent_id
        steps = chunks[1: -1]
        path = list(map(lambda s: eval(s), steps))
    return path


def dxdy2action(dxdy):
    if dxdy == (0, 0):
        return 0  # stop
    elif dxdy == (-1, 0):
        return 1  # up
    elif dxdy == (0, 1):
        return 2  # right
    elif dxdy == (1, 0):
        return 3  # down
    elif dxdy == (0, -1):
        return 4  # left
