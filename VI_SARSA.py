import sys, time, argparse
import gym
import numpy as np
from tqdm import tqdm
from lib.common_utils import TabularUtils
from lib.regEnvs import *

class Tabular_DP:
    def __init__(self, args):
        self.env = args.env # 
        self.gamma = 0.99 # discount factor
        self.theta = 1e-5 # convergence threshold
        self.max_iterations = 1000
        self.nA = self.env.action_space.n # 4
        self.nS = self.env.observation_space.n # 16 or 64

    def compute_q_value_cur_state(self, s, value_func):
        q_s = np.zeros(self.nA)
        probability_matrix = self.env.P

        # TODO: write your code here
        # return: q_value for state s [float array with shape (nA)]
        for action_index in range(self.nA):
            transitions = probability_matrix[s][action_index]
            for t in transitions:
                prob, new_s, reward, _ = t
                q_s[action_index] += prob * (reward + self.gamma * value_func[new_s])
        return q_s

    def policy_iteration(self):
        value_func = np.zeros(self.nS)
        policy = np.zeros(self.nS)
        probability_matrix = self.env.P
        while True:
            delta = self.theta + 1
            while delta > self.theta:
                delta = 0
                for state_index in range(self.nS):
                    previous_value_state = value_func[state_index]
                    action = policy[state_index]
                    transitions = probability_matrix[state_index][action]
                    new_value = 0
                    for t in transitions:
                        prob, new_s, reward, _ = t
                        new_value += prob * (reward + self.gamma * value_func[new_s])
                    value_func[state_index] = new_value
                    delta = max(delta, abs(previous_value_state - value_func[state_index]))
            policy_stable = True
            for state_index in range(self.nS):
                old_action = policy[state_index]
                q_values = self.compute_q_value_cur_state(state_index, value_func)
                policy[state_index] = np.argmax(q_values)
                if old_action != policy[state_index]:
                    policy_stable = False
            if policy_stable:
                break
        return value_func, TabularUtils(self.env).deterministic_policy_to_onehot_policy(policy.astype(int))

    def value_iteration(self):
        value_func = np.zeros(self.nS)
        policy_optimal = np.zeros([self.nS, self.nA])
        
        # TODO: write your code here
        # return1 V_optimal [float array with shape (nS)]
        # return2 policy_optimal [one hot array with shape (nS x nA)]
        delta = 1
        num_iter = 0
        while delta > self.theta and num_iter < self.max_iterations:
            delta = 0
            for state_index in range(self.nS):
                previous_value_state = value_func[state_index]
                q_values = self.compute_q_value_cur_state(state_index, value_func)
                value_func[state_index] = np.max(q_values)
                policy_optimal[state_index, :] = TabularUtils(self.env).action_to_onehot(
                    np.argmax(q_values))
                delta = max(delta, abs(previous_value_state - value_func[state_index]))
            num_iter += 1

        return value_func, policy_optimal



class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes = 10000
        self.gamma = 0.99
        self.alpha = 0.05
        self.env_nA = self.env.action_space.n
        self.env_nS = self.env.observation_space.n
        self.tabularUtils = TabularUtils(self.env)


    def q_learning(self):
        Q = np.zeros((self.env_nS, self.env_nA))
        epsilon = 0.01
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = tabularUtils.epsilon_greedy_policy(Q[state], epsilon)
                next_state, reward, done, _ = self.env.step(action)
                best_next_action = np.argmax(Q[next_state])
                Q[state][action] = (1 - self.alpha) * Q[state][action] + self.alpha * (
                        reward + self.gamma * Q[next_state][best_next_action])
                state = next_state
        optimal_policy = tabularUtils.deterministic_policy_to_onehot_policy(np.argmax(Q, axis=1))
        return Q, optimal_policy

    

    def sarsa(self):
        Q = np.zeros((self.env_nS, self.env_nA))
        # TODO: write your code here
        # return1 Q value [float array with shape (nS x nA)]
        # return2 salsa_policy [one hot array with shape (nS x nA)]
        epsilon = 0.01
        for episode in range(self.num_episodes):
            s = self.env.reset()
            a = tabularUtils.epsilon_greedy_policy(Q[s, :], epsilon)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(a)
                next_action = tabularUtils.epsilon_greedy_policy(Q[next_state, :], epsilon)
                Q[s, a] = Q[s, a] + self.alpha * (
                        reward + self.gamma * Q[next_state, next_action] - Q[s, a])
                s = next_state
                a = next_action
        salsa_policy = tabularUtils.deterministic_policy_to_onehot_policy(np.argmax(Q, axis=1))
        return Q, salsa_policy


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_size', dest='map_size', type=int, default=4,  # Default map size is 4x4
                        choices=[4, 8], help="Specify the map size: 4 or 8.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    args.map_size = 8
    if args.map_size == 4:
        args.env_name = "FrozenLake-Deterministic-v1"
    elif args.map_size == 8:
        args.env_name = "FrozenLake-Deterministic-8x8-v1"
    args.env = gym.make(args.env_name)
    tabularUtils = TabularUtils(args.env)
    
    # example dummy policies
    if args.map_size == 4:
        dummy_policy = np.array([1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 3, 1, 2, 2, 3, 0])
    elif args.map_size == 8:
        dummy_policy = np.array([2, 2, 2, 2, 2, 2, 2, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0])
    one_not_dummy_policy = tabularUtils.deterministic_policy_to_onehot_policy(dummy_policy)
    # render
    # tabularUtils.render(one_not_dummy_policy)

    # test value iteration
    dp = Tabular_DP(args)
    value_function_vi, policy_optimal_vi = dp.value_iteration()
    print(f'Value function of {args.map_size} x {args.map_size} environment using Value Iteration:', value_function_vi)
    print(f'Policies of {args.map_size} x {args.map_size} environment using Value Iteration:',
          tabularUtils.onehot_policy_to_deterministic_policy(policy_optimal_vi))

    value_function_pi, policy_optimal_pi = dp.policy_iteration()
    print(f'Value function of {args.map_size} x {args.map_size} environment using Policy Iteration:', value_function_pi)
    print(f'Policies of {args.map_size} x {args.map_size} environment using Policy Iteration:',
          tabularUtils.onehot_policy_to_deterministic_policy(policy_optimal_pi))
    
    # test SARSA
    td = Tabular_TD(args)
    q_function_sarsa, policy_optimal_sarsa = td.sarsa()
    print(f'Q function of {args.map_size} x {args.map_size} environment using SARSA:', q_function_sarsa)
    print(f'Policies of {args.map_size} x {args.map_size} environment using SARSA',
          tabularUtils.onehot_policy_to_deterministic_policy(policy_optimal_sarsa))

    q_function_q_learning, policy_optimal_q_learning = td.q_learning()
    print(f'Q function of {args.map_size} x {args.map_size} environment using Q Learning:', q_function_q_learning)
    print(f'Policies of {args.map_size} x {args.map_size} environment using Q Learning',
          tabularUtils.onehot_policy_to_deterministic_policy(policy_optimal_q_learning))

    # render a video
    tabularUtils.render(policy_optimal_vi)
    tabularUtils.render(policy_optimal_pi)
    tabularUtils.render(policy_optimal_sarsa)
    tabularUtils.render(policy_optimal_q_learning)


