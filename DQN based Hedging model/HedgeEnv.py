import numpy as np
from RoughVolOptionPriceSim import generate_sim


class HedgeEnv():

    def __init__(self, num_paths, num_time):
        self.num_paths = num_paths
        self.num_time = num_time
        self.num_sold_opt = 100
        self.rho = -0.02
        self.T = 1
        self.alpha = -0.43
        self.eta = 1.0
        self.xi = 1.0
        self.S0 = 100
        self.K = 100
        self.r = 0.5
        # self.seed = 0
        self.dt = self.T / num_time
        self.num_states = 6  # ?????????
        self.num_actions = self.num_sold_opt + 1
        self.initialize_paths = False
        self.kappa = 0.001  # balance of profit and risk
        self.idx_time = 0
        self.idx_path = -1
        self.state = None  # ?

    def __generate_paths(self):
        self.T_list, self.V, self.S, self.C, self.delta = generate_sim(self.num_paths, self.num_time, self.rho, self.T,
                                                                       self.alpha, self.eta, self.xi, self.S0, self.K,
                                                                       self.r)
        self.idx_path = -1

    def __get_state_without_num_stocks(self, i_time, j_path):
        t = self.T_list[i_time]
        V = self.V[i_time, j_path]
        S = self.S[i_time, j_path]
        C = self.C[i_time, j_path]
        delta = self.delta[i_time, j_path] * self.num_sold_opt  # multiplied by num options
        num_stock = 0
        return np.array([t, V, S, C, delta, num_stock])  # state

    def clear_all_paths(self):
        self.initialize_paths = False

    def reset(self):
        if not self.initialize_paths:
            self.__generate_paths()
            self.initialize_paths = True

        self.idx_path = (self.idx_path + 1) % self.num_paths  # self.idx_path set -1 in generator
        self.idx_time = 0

        state = self.__get_state_without_num_stocks(self.idx_time, self.idx_path)
        self.state = state

        return state

    def step(self, action):
        if self.idx_time > self.num_time:
            n_state = None
            reward = np.nan
            done = True

        elif self.idx_time == self.num_time:
            n_state = None
            reward = self.__get_reward(n_state)
            done = True

        else:
            self.idx_time += 1
            n_state = self.__get_state_without_num_stocks(self.idx_time, self.idx_path)
            n_state[5] = action  # num of stocks is updated.
            reward = self.__get_reward(n_state)
            done = False

        self.state = n_state

        return n_state, reward, done

    def __get_reward(self, n_state=None):
        gamma = 0.99
        if n_state is None or self.state[0] == self.T:
            reward = self.num_sold_opt * (np.exp(self.r * self.T) - 1) * self.C[0, self.idx_path]
            reward = reward / np.power(gamma, self.T)
            return reward

        t1 = n_state[0]
        t0 = self.state[0]

        V1 = n_state[1]
        V0 = self.state[1]

        S1 = n_state[2]
        S0 = self.state[2]

        C1 = n_state[3]
        C0 = self.state[3]

        d1 = n_state[4]  # = delta per one option * num sold options
        d0 = self.state[4]  # = delta per one option * num sold options

        nS1 = n_state[5]
        nS0 = self.state[5]

        reward = nS1 * S1 - nS0 * S0 - self.num_sold_opt * (C1 - C0) - (nS1 - nS0) * S0 * np.exp(self.r * (self.T - t0))

        if self.kappa > 0:
            var = np.sqrt(V0) * S0 * (nS1 - d0)
            var = var ** 2 * self.dt
            reward = reward - self.kappa * var / 2

        reward = reward / np.power(gamma, t0)

        return reward
