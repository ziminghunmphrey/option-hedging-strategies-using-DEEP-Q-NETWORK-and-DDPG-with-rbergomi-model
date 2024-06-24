import numpy as np



class Trainer:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run(self, num_episodes, num_time):

        hists = None
        self.env.clear_all_paths()

        for episode in range(num_episodes):
            state = self.env.reset()
            if hists is None:
                hists = np.zeros(shape=(num_episodes, num_time + 1, len(state)))

            for step in range(num_time + 1):
                action = self.agent.decide_action(state)
                n_state, reward, done = self.env.step(action)

                hists[episode, step, :-1] = state

                if done:
                    break
                else:
                    state = n_state

        return hists

    def train(self, num_episodes, num_time):
        gamma = 0.99

        list_reward = []

        for episode in range(num_episodes):

            state = self.env.reset()
            sum_reward = 0

            for step in range(num_time + 1):
                action = self.agent.decide_action(state, episode=episode)
                n_state, reward, done = self.env.step(action)
                sum_reward = sum_reward + reward * np.power(gamma, step)

                self.agent.memory.push(state, action, n_state, reward)
                self.agent.replay()

                if done:
                    break
                else:
                    state = n_state

            list_reward.append(sum_reward)
            if episode % 10 == 0:
                print("Episode {}:  Reward = {}".format(episode, np.round(list_reward[episode], 2)))

        return list_reward
