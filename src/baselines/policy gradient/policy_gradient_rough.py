#!/user/bw2762/.conda/envs/testbed_2/bin/python
import time
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.optim as optim

class MLP_Policy(nn.Module):
    """
    MLP Policy network. Outputs are logits.
    """
    def __init__(self, in_dim, out_dim, hidden_size=64):
        super(MLP_Policy, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MLP(nn.Module):
    """MLP as toy UQ"""
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class MLP_experiment():
    """
    experiment for training mlp (UQ)
    """

    def __init__(self,in_dim,out_dim,num_epochs,learning_rate, model_seed):

        self.model = MLP(in_dim,out_dim)
        for param in self.model.parameters():
          torch.nn.init.uniform_(param, a=-1, b=1) #init the parameter for the mlp
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9) # set sgd optimizer



    def step(self,batch):
        """MLP training Step"""
        for i in range(self.num_epochs):
            features,labels = batch.tensors
            outputs = self.model(features)
            outputs = torch.sigmoid(outputs)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def get_flat_params(self):
        """
        get flattened mlp params
        """
        parameters = [p.data for p in self.model.parameters()]
        flat_parameters = torch.cat([p.view(-1) for p in parameters])
        return flat_parameters

    def predict(self,x):
        return self.model.forward(x)

class toy_MLP_ENV(gym.Env):

    T = 1
    in_dim = 20
    out_dim = 1
    num_epochs = 10 #number of epochs for each batch in MLP experiment

    def __init__(self,dataset,batch_size,learning_rate,seed_policy = 0, seed_model=0):

        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.t = 0
        self.seed_model = seed_model
        # num_dataset = len(dataset)
        self.experiment = MLP_experiment(self.in_dim,self.out_dim,self.num_epochs,self.learning_rate, self.seed_model)
        self.init_model = copy.deepcopy(self.experiment.model)
        #self.action_space = spaces.MultiBinary(num_dataset) #gym settings currently not needed
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(num_dataset,)) #gym settings currently not needed
        print("INITIALIZED")

    def reset(self, seed=None, options=None):
        #super().reset(seed=seed, options=options)
        self.t = 0
        self.experiment = MLP_experiment(self.in_dim,self.out_dim,self.num_epochs,self.learning_rate, self.seed_model)
        self.experiment.model.load_state_dict(self.init_model.state_dict())
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return None

    def _get_info(self):
        return None

    def _get_var(self):
        """obj is sum of squares"""
        params = self.experiment.get_flat_params()
        sum_of_squares = params.square().sum()
        return sum_of_squares

    def step(self, action):
        """environment step"""
        batch = self.dataset[action]
        batch = TensorDataset(batch[0],batch[1])
        self.experiment.step(batch)
        observation = self._get_obs() # not needed
        reward = self._get_var()

        terminated = False # check if it should terminate (we currently just have 1 step)
        self.t += 1
        if self.t >= self.T:
            terminated = True

        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

def train(env,policy, optimizer, num_episode, features):
    reward_pool = []

    steps = 0

    for episode in tqdm(range(num_episode)):
        state = env.reset() # reset env, state currenly not needed
        #env.render()

        for t in range(1):
            w = policy(features)
            w = w.squeeze()
            prob = F.softmax(w, dim=0)   
            
            loss_temp = []
            for j in range(10):
                batch_ind = torch.multinomial(prob, env.batch_size, replacement=True)
                log_pr = torch.log(prob[batch_ind]).sum()
                action = batch_ind

                next_state, reward, done, truncated, info = env.step(action) # env step, uq update
                loss_temp.append(log_pr*reward)
                env.reset()

            loss = torch.stack(loss_temp).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                          
            env.render()

            reward_pool.append(reward)

            steps += 1

            if done:
                break

    return reward_pool

### set learning parameters
learning_rate_PG = 1e-4
learning_rate_MLP = 1e-1
num_episode = 1000

batch_size = 2
n = 20
feature_dim = 20

in_dim = feature_dim
out_dim = 1
torch.manual_seed(1)

# generate dataset
features = torch.randn(n,feature_dim)
# features = torch.clamp(features, -1, 1)
labels = torch.randint(0, 2, (n,))
features_tensor = torch.tensor(features, dtype=torch.float32)
print(features_tensor)
labels_tensor = torch.tensor(labels, dtype=torch.float32)
print(labels)
train_dataset = TensorDataset(features_tensor, labels_tensor)
print(train_dataset)



# init env and policy
env = toy_MLP_ENV(train_dataset,batch_size,learning_rate_MLP,0,0)
policy = MLP_Policy(in_dim,out_dim)
optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate_PG)

# train policy
reward_pool = train(env,policy,optimizer,num_episode,features_tensor)

# visualize training procedure
data_series = pd.Series(reward_pool)
# rolling_mean = data_series
rolling_mean = data_series.rolling(window=50).mean()
plt.plot(rolling_mean)
plt.savefig('pg_test_rough.jpg')