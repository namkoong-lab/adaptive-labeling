#!/user/bw2762/.conda/envs/testbed_2/bin/python

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import jax
import jax.numpy as jnp

from enn import networks
from enn import datasets
from enn import supervised
from enn import utils

from enn_ENV_demo import UQ_epinet_ENV
from RL_utils import split_dataset

from neural_testbed_test_1.neural_testbed.RL_stuff.factories_epinet_v2 import make_agent_v2, EpinetConfig_v2
from neural_testbed_test_1.neural_testbed.RL_stuff.enn_agents_v2 import extract_enn_sampler_v2
from neural_testbed_test_1.neural_testbed.UQ_data.data_modules_2 import generate_problem_v2

#Hyperparameters
learning_rate = 0.01
# gamma = 0.98
gamma = 1

# num_episode = 5000
num_episode = 50
batch_size = 32

config = EpinetConfig_v2(
  index_dim = 8,  # Index dimension
  l2_weight_decay = 0.2,  # Weight decay
  distribution = 'none',  # Bootstrap distribution
  prior_scale = 0.3,  # Scale of the additive prior function
  prior_scale_epi = 0.,  # Scale of the epinet prior function
  prior_loss_freq = 100_000,  # Prior loss frequency
  hidden_sizes = (50, 50),  # Hidden sizes for the neural network
#   num_batches = 1_000,  # Number of SGD steps
  epi_hiddens = (15, 15),  # Hidden sizes in epinet
  add_hiddens = (5, 5),  # Hidden sizes in additive prior
  seed = 0,  # Initialization seed
  override_index_samples = None,  # Set SGD training index samples
  learning_rate = 1e-3,  # Learning rate for adam optimizer
)

dataset_name = 'eicu'
path_train = '/user/bw2762/UQ_implementation_shared/datasets/eicu_train_final.csv'
path_test = '/user/bw2762/UQ_implementation_shared/datasets/eicu_test_final.csv'
label_name = 'EVENT_LABEL'
num_classes = 2
tau =10
seed = 1
temperature = 0.01
noise_std = 1.
sampler_type = 'global'

problem_eicu = generate_problem_v2(path_train,path_test,label_name,dataset_name,sampler_type,num_classes,tau,seed,temperature,noise_std)

train_dataset = problem_eicu.train_data

key = jax.random.PRNGKey(0)
train_data_eicu, calib_data_eicu, first_batch_eicu = split_dataset(key, train_dataset,train_frac=0.8,calib_frac=0.001)

# n_samples = len(train_data.x)
# subset_ind = jax.random.choice(jax.random.PRNGKey(0),n_samples,(3000,),replace = False)
# first_batch_eicu = subset_data = datasets.ArrayBatch(x=train_data.x[subset_ind], y=train_data.y[subset_ind])

first_batch = first_batch_eicu
dataset = train_data_eicu
calibration_dataset = calib_data_eicu
problem = problem_eicu
agent_config = config
seed = 0
batch_num = 300

env = UQ_epinet_ENV(first_batch,dataset,calibration_dataset,problem,agent_config,batch_num)
# env = UQ_epinet_ENV()
# env = env(first_batch,dataset,calibration_dataset,problem,agent_config,batch_num)
print(env.observation_space.shape)
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

print(state_space)
print(action_space)

def plot_durations(episode_durations):
    plt.ion()
    plt.figure(2)
    plt.clf()
    duration_t = torch.FloatTensor(episode_durations)
    plt.title('Training')
    plt.xlabel('Episodes')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy())

    if len(duration_t) >= 100:
        means = duration_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.00001)

class Policy(nn.Module):

    def __init__(self,state_space,action_space):
        super(Policy, self).__init__()

        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        #x = F.dropout(x, 0.5)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=-1)

        return x

policy = Policy(state_space,action_space)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

state = env.reset()
state = np.array(state)
state = torch.from_numpy(state).float()
state = Variable(state)
action = policy(state)
action = action.data.numpy().astype('float32')
next_state, reward, done, truncated, info = env.step(action)


# def train():

#     episode_durations = []
#     #Batch_history
#     state_pool = []
#     action_pool = []
#     reward_pool = []
#     steps = 0

#     for episode in range(num_episode):
#         state = env.reset()

#         # state = env._get_obs()
        
#         state = np.array(state)
#         state = torch.from_numpy(state).float()
#         state = Variable(state)

#         env.render()

#         for t in count():
#             # probs = policy(state)
#             # c = Categorical(probs)
#             # action = c.sample()

#             action = policy(state)

#             action = action.data.numpy().astype('float32')
#             next_state, reward, done, truncated, info = env.step(action)
#             reward = 0 if done else reward # correct the reward
#             env.render()

#             state_pool.append(state)
#             action_pool.append(float(action))
#             reward_pool.append(reward)

#             state = next_state
#             state = torch.from_numpy(state).float()
#             state = Variable(state)

#             steps += 1

#             if done:
#                 episode_durations.append(t+1)
#                 plot_durations(episode_durations)
#                 break

#         # update policy
#         if episode >0 and episode % batch_size == 0:

#             r = 0
#             '''
#             for i in reversed(range(steps)):
#                 if reward_pool[i] == 0:
#                     running_add = 0
#                 else:
#                     running_add = running_add * gamma +reward_pool[i]
#                     reward_pool[i] = running_add
#             '''
#             for i in reversed(range(steps)):
#                 if reward_pool[i] == 0:
#                     r = 0
#                 else:
#                     r = r * gamma + reward_pool[i]
#                     reward_pool[i] = r

#             #Normalize reward
#             reward_mean = np.mean(reward_pool)
#             reward_std = np.std(reward_pool)
#             reward_pool = (reward_pool-reward_mean)/reward_std

#             #gradiend desent
#             optimizer.zero_grad()

#             for i in range(steps):
#                 state = state_pool[i]
#                 action = Variable(torch.FloatTensor([action_pool[i]]))
#                 reward = reward_pool[i]

#                 probs = policy(state)
#                 c = Categorical(probs)

#                 loss = -c.log_prob(action) * reward
#                 loss.backward()

#             optimizer.step()

#             # clear the batch pool
#             state_pool = []
#             action_pool = []
#             reward_pool = []
#             steps = 0

# train()