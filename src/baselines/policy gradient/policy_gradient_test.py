#!/user/bw2762/.conda/envs/testbed_2/bin/python

from tqdm import tqdm

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.is_available())

from mlp_ENV_demo import toy_MLP_ENV
from RL_utils import split_dataset

# from neural_testbed_test_1.neural_testbed.RL_stuff.factories_epinet_v2 import make_agent_v2, EpinetConfig_v2
# from neural_testbed_test_1.neural_testbed.RL_stuff.enn_agents_v2 import extract_enn_sampler_v2
from neural_testbed_test_1.neural_testbed.UQ_data.data_modules_2 import generate_problem_v2

#Hyperparameters
learning_rate = 0.01
# gamma = 0.98
gamma = 1

# num_episode = 5000
num_episode = 5000
batch_size = 32

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


train_data_eicu, calib_data_eicu, first_batch_eicu = split_dataset(train_dataset,train_frac=0.8,calib_frac=0.001)

first_batch = first_batch_eicu
dataset = train_data_eicu
calibration_dataset = calib_data_eicu
problem = problem_eicu
# agent_config = config
seed = 0
batch_num = 500

env = toy_MLP_ENV(first_batch,dataset,batch_num)

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# def plot_durations(episode_durations):
#     plt.ion()
#     plt.figure(2)
#     plt.clf()
#     duration_t = torch.FloatTensor(episode_durations)
#     plt.title('Training')
#     plt.xlabel('Episodes')
#     plt.ylabel('Duration')
#     plt.plot(duration_t.numpy())

#     if len(duration_t) >= 100:
#         means = duration_t.unfold(0,100,1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.00001)


class Policy(nn.Module):
    """
    Policy network, input state then get mean and std. Actions are sampled from Normal(mean,std).
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean_layer = nn.Linear(hidden_size, action_size)

        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)

        return mean, std




#INITIALIZATION
policy = Policy(state_space,action_space)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
state = env.reset()
state = Variable(state)
# action = policy(state)
mean, std = policy(state)
action = torch.normal(mean, std)
next_state, reward, done, truncated, info = env.step(action)


def train():

    # episode_durations = []

    #Batch_history
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    for episode in tqdm(range(num_episode), desc="Training Progress"):
        state = env.reset()
        state = Variable(state)

        env.render()

        for t in count():
            mean, std = policy(state)
            action = torch.normal(mean, std)
            next_state, reward, done, truncated, info = env.step(action)
            # reward = 0 if done else reward 

            env.render()

            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)

            state = next_state
            state = Variable(state)

            steps += 1

            if done:
                # episode_durations.append(t+1)
                # plot_durations(episode_durations)
                break

        # update policy
        if episode >0 and episode % batch_size == 0:

            #Discounted reward
            r = 0
            '''
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma +reward_pool[i]
                    reward_pool[i] = running_add
            '''
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    r = 0
                else:
                    r = r * gamma + reward_pool[i]
                    reward_pool[i] = r

            #Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            reward_pool = (reward_pool-reward_mean)/reward_std

            #gradiend desent
            optimizer.zero_grad()

            policy_loss = 0

            for i in range(steps):
                state = state_pool[i]
                mean, std = policy(state)
                action = action_pool[i]
                # action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]
                
                normal_dist = torch.distributions.Normal(mean,std)
                action_log_prob = normal_dist.log_prob(action)
                action_log_prob = action_log_prob.sum()
                loss = - action_log_prob * reward
                policy_loss += loss

                # loss = -c.log_prob(action) * reward
                # loss.backward()

            # print(policy_loss)
            policy_loss.backward()
            optimizer.step()

            print(env._get_var()) # check

            # clear the batch pool
            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


train()