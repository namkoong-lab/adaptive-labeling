#!/user/bw2762/.conda/envs/testbed_2/bin/python

import argparse
import gymnasium as gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

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

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


# env = gym.make('CartPole-v0')
# env.seed(args.seed)
# torch.manual_seed(args.seed)


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

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.affine2 = nn.Linear(128, 2)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = F.relu(self.affine1(x))
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)


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
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()

        state = np.array(state)
        state = torch.from_numpy(state).float()
        state = Variable(state)

        for t in range(10000):  # Don't infinite loop while learning
            # action = select_action(state)

            state = np.array(state)
            state = torch.from_numpy(state).float()
            state = Variable(state)
            action = policy(state)
            action = action.data.numpy().astype('float32')
            print(type(action))
            state, reward, done, _ , info = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()