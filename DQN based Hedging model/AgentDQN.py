import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from ReplayMemory import ReplayMemory


class AgentDQN:

    def __init__(self, num_states, num_actions):

        self.num_states = num_states
        self.num_actions = num_actions
        self.freq_update_target = 5  # set frequency of updating target
        self.count_replay = 0
        self.memory = ReplayMemory(10000)  # set capacity

        # Construct a neural network
        self.model = models.Sequential()
        self.model.add(layers.Dense(input_shape=(num_states,), units=128, activation='relu'))
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(num_actions))
        self.model.summary()
        # Set how to train the model
        self.model.compile(loss='mse', optimizer=optimizers.SGD())
        self._target_model = models.clone_model(self.model)

    def save(self, file_name):
        self.model.save('./save_model' + file_name)

    def load(self, file_name):
        self.model = models.load_model('./save_model' + file_name)
        self._target_model = models.clone_model(self.model)

    def replay(self):
        batch_size = 32
        if self.memory.index < batch_size:
            return

        # Make mini batch
        transitions = self.memory.sample(batch_size)

        states = np.array(transitions[0, :])
        actions = np.array(transitions[1, :])
        n_states = np.array(transitions[2, :])
        rewards = np.array(transitions[3, :])

        X = np.zeros(shape=(n_states.shape[0], self.num_states))
        Y = np.zeros(shape=(states.shape[0], self.num_states))
        for i, n_s in enumerate(n_states):
            X[i, :] = n_s
        n_q = np.max(self._target_model.predict(X), axis=1)

        # Make targets for regression
        gamma = 0.99
        for i, s in enumerate(states):
            Y[i, :] = s
        states = Y.tolist()
        q = self.model.predict(states)
        for i, n_s in enumerate(n_states):
            r = rewards[i]
            a = actions[i]
            if n_s is not None:
                r = r + gamma * n_q[i]
            q[i, a] = r

        # Update weight parameters
        q = q.tolist()
        train_set = tf.data.Dataset.from_tensor_slices((states, q)).batch(batch_size)
        for x, y in train_set:
            self.model.train_on_batch(x, y)

        self.count_replay += 1
        # if self.count_replay % self.freq_update_target == 0:
        swa_weight = ((self.count_replay - 1) * np.array(self._target_model.get_weights()) +
                      np.array(self.model.get_weights())) / self.count_replay
        self._target_model.set_weights(swa_weight.tolist())

    def decide_action(self, state, episode=None):

        if episode is not None:
            epsilon = 0.5 * (1 / (episode + 1))
        else:
            epsilon = 0

        if np.random.uniform(0, 1) >= epsilon:
            state = np.array(state).reshape(1, -1)
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = np.random.randint(0, self.num_actions)

        return action
