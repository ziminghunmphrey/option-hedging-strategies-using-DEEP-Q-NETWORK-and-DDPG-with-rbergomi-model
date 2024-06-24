import numpy as np


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.zeros([4, 1])
        self.index = 0

    def push(self, state, action, state_next, reward):
        if self.index < self.capacity:
            self.memory = np.hstack((self.memory, np.array([state, action, state_next, reward]).reshape([4, 1])))
            self.index = self.index + 1

    def sample(self, batch_size):
        choice = np.random.choice(range(1, self.memory.shape[1]), batch_size, replace=False)
        return self.memory[:, choice]
