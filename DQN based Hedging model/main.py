from AgentDQN import AgentDQN
from HedgeEnv import HedgeEnv
from Trainer import Trainer

env = HedgeEnv(300, 200)
num_states = env.num_states
num_actions = env.num_actions
num_time = env.num_time

agent = AgentDQN(num_states, num_actions)

trainer = Trainer(env, agent)

train_num_episodes = 300
list_reward = trainer.train(train_num_episodes, num_time)
run_num_episodes = 100
hists = trainer.train(run_num_episodes,num_time)