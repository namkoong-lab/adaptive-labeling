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

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.optim as optim

import math

class CustomizableGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, base_kernel, likelihood):
        super(CustomizableGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = base_kernel
        self.likelihood = likelihood

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class ConstantValueNetwork(nn.Module):
    def __init__(self, constant_value=1.0, output_size=1):
        super(ConstantValueNetwork, self).__init__()
        # Define the constant value and output size
        self.constant_value = nn.Parameter(torch.tensor([constant_value]*output_size), requires_grad=False)
        self.output_size = output_size

    def forward(self, x):
        # x is your input tensor. Its value is ignored in this model.
        # Return a 1-D tensor with the constant value for each item in the batch.
        batch_size = x.size(0)  # Get the batch size from the input
        return self.constant_value.expand(batch_size, self.output_size)

### Adapting L_2 loss for the GP pipeine

def var_l2_loss_estimator(model, test_x, Predictor, device, para):
    #z_dim = para['z_dim']

    #N_iter =  para['N_iter']
    #if_print =  para['if_print']
    #seed = para['seed_var_l2']
    #torch.manual_seed(seed)

    N_iter =  100
    # seed = 0
    # torch.manual_seed(seed)

    #res  = torch.empty((0), dtype=torch.float32, device=device)
    #res_square  = torch.empty((0), dtype=torch.float32, device=device)



    latent_posterior = model(test_x)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([N_iter]))
    #print("latent_posterior_sample:", latent_posterior_sample)
    #print("latent_posterior_sample.shape:", latent_posterior_sample.shape)
    prediction = Predictor(test_x).squeeze()
    #print("prediction:", prediction)
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample, prediction))
    #print("L_2_loss_each_point:", L_2_loss_each_point)
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    #print("L_2_loss_each_f_size:", L_2_loss_each_f.shape)
    #print("L_2_loss_each_f:",L_2_loss_each_f)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    # print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    # print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_variance

def l2_loss(test_x, test_y, Predictor, device):
    prediction = Predictor(test_x).squeeze()
    #print("prediction:", prediction)
    #print("test_y:", test_y)
    diff_square = torch.square(torch.subtract(test_y, prediction))
    #print("diff_square:", diff_square)
    return torch.mean(diff_square)

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

class new_MLP_Policy(nn.Module):
    """
    MLP Policy network. Outputs are logits.
    """
    def __init__(self, pool_size):
        super(new_MLP_Policy, self).__init__()
        reciprocal_size_value =  math.log(1.0 / tensor_size)
        # Now compute the logarithm of the reciprocal
        NN_weights = torch.full([tensor_size], reciprocal_size_value, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x      
    
# class MLP(nn.Module):
#     """MLP as toy UQ"""
#     def __init__(self,input_dim,output_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         # self.fc2 = nn.Linear(2, 2)
#         self.fc3 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         # x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

class GP_experiment():
    """
    experiment for training GP (UQ)
    """

    def __init__(self,model,train_x,train_y):

        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        # self.model = MLP(in_dim,out_dim)
        # for param in self.model.parameters():
        #   torch.nn.init.uniform_(param, a=-1, b=1) #init the parameter for the mlp
        # self.criterion = nn.BCELoss()
        # self.learning_rate = learning_rate
        # self.num_epochs = num_epochs
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9) # set sgd optimizer



    def step(self,x,y):
        """GP training Step"""
        # features,labels = batch.tensors
        new_train_x = torch.cat([self.train_x,x])
        new_train_y = torch.cat([self.train_y,y])
        self.model.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)


    # def get_flat_params(self):
    #     """
    #     get flattened mlp params
    #     """
    #     parameters = [p.data for p in self.model.parameters()]
    #     flat_parameters = torch.cat([p.view(-1) for p in parameters])
    #     return flat_parameters

    # def predict(self,x):
    #     return self.model.forward(x)

class toy_GP_ENV(gym.Env):

    T = 1
    # in_dim = 20
    # out_dim = 1
    # num_epochs = 10 #number of epochs for each batch in MLP experiment

    def __init__(self,train_x,train_y,test_x,pool_x,pool_y,model,Predictor,batch_size,seed_policy = 0, seed_model=0):

        super().__init__()
        self.batch_size = batch_size
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.pool_x = pool_x
        self.pool_y = pool_y
        self.model = model
        self.Predictor = Predictor
        # self.learning_rate = learning_rate
        self.t = 0
        self.seed_model = seed_model
        # num_dataset = len(dataset)
        self.experiment = GP_experiment(self.model,self.train_x,self.train_y)
        self.init_model = copy.deepcopy(self.experiment.model)
        #self.action_space = spaces.MultiBinary(num_dataset) #gym settings currently not needed
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(num_dataset,)) #gym settings currently not needed
        print("INITIALIZED")

    # def reset(self, seed=None, options=None):
    #     #super().reset(seed=seed, options=options)
    #     self.t = 0
    #     self.experiment = GP_experiment(self.model,self.train_x,self.train_y)
    #     self.experiment.model.load_state_dict(self.init_model.state_dict())
    #     obs = self._get_obs()
    #     return obs

    def reset(self, seed=None, options=None):
        return None

    def _get_obs(self):
        return None

    def _get_info(self):
        return None

    # def _get_loss(self):


    # def _get_var(self):
    #     """obj is sum of squares"""
    #     params = self.experiment.get_flat_params()
    #     sum_of_squares = params.square().sum()
    #     return sum_of_squares

    def step(self, action):
        """environment step"""
        x = self.pool_x[action]
        y = self.pool_y[action]
        # batch = self.dataset[action]
        # batch = TensorDataset(batch[0],batch[1])
        self.experiment.step(x,y)
        observation = self._get_obs() # not needed
        loss = var_l2_loss_estimator(self.experiment.model, self.test_x, self.Predictor, None, None)
        # loss = -loss

        terminated = False # check if it should terminate (we currently just have 1 step)
        self.t += 1
        if self.t >= self.T:
            terminated = True

        truncated = False
        terminated = True
        info = self._get_info()

        return observation, loss, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

def train(env, w, optimizer, num_episode, features):
    loss_pool = []

    steps = 0

    for episode in tqdm(range(num_episode)):
        state = env.reset() # reset env, state currenly not needed
        #env.render()

        for t in range(1):
            # w = policy(features)
            w = w.squeeze()
            prob = F.softmax(w, dim=0)   
            print(prob)
            
            loss_temp = []
            for j in range(1000):
                batch_ind = torch.multinomial(prob, env.batch_size, replacement=False)
                log_pr = (torch.log(prob[batch_ind])).sum()
                for i in range(env.batch_size):
                    log_pr = log_pr- torch.log(1 - prob[batch_ind[:i]].sum())
                action = batch_ind
                # print(action)

                next_state, loss, done, truncated, info = env.step(action) # env step, uq update
                loss_temp.append(log_pr*loss)
                env.reset()

            avg_loss = torch.stack(loss_temp).mean()
            print(avg_loss)

            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
                          
            env.render()

            loss_pool.append(avg_loss.detach().numpy())

            steps += 1

            if done:
                break

    return loss_pool


num_init_train_samples = 20
num_pool_samples = 5
num_test_samples = 20

input_dim = 1

init_train_x = torch.rand((num_init_train_samples, input_dim))*50.0
test_x_1 = torch.rand((num_test_samples, input_dim))*50.0
test_x_2 = 75.0 + torch.rand((num_test_samples, input_dim))*50.0
test_x_3 = 175.0 + torch.rand((num_test_samples, input_dim))*50.0
test_x = torch.cat([test_x_1,test_x_2,test_x_3])
pool_x_1 = 24 + torch.rand((num_pool_samples, input_dim))*2
pool_x_2 = 99 + torch.rand((num_pool_samples, input_dim))*2
pool_x_3 = 199 + torch.rand((num_pool_samples, input_dim))*2
pool_x = torch.cat([pool_x_1,pool_x_2,pool_x_3])

y = torch.zeros(num_init_train_samples+3*num_pool_samples+3*num_test_samples)

init_train_x_numpy = init_train_x.numpy()
init_train_y = torch.zeros(init_train_x.size(0))
test_x_numpy = test_x.numpy()
test_y = torch.ones(test_x.size(0))
pool_x_numpy = pool_x.numpy()
pool_y = torch.empty(pool_x.size(0)).fill_(0.5)


# plt.scatter(init_train_x_numpy, init_train_y.numpy(), s=20, label='train')
# plt.scatter(test_x_numpy, test_y.numpy(), s=20, label='test')
# plt.scatter(pool_x_numpy, pool_y.numpy(), s=20, label='pool')

# plt.yticks([])  # Hide y-axis ticks
# plt.xlabel('X values')
# plt.legend()
# plt.title('Distribution of X values along a real line')
# plt.savefig('pg_gpr_example.jpg')
# plt.show()

x = torch.cat([init_train_x,test_x,pool_x])

# Define parameters for the model
mean_constant = 0.0  # Mean of the GP
length_scale = 25.0   # Length scale of the RBF kernel
noise_std = 0.01     # Standard deviation of the noise
output_scale = 0.6931471824645996

mean_module = gpytorch.means.ConstantMean()
base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()

mean_module.constant = mean_constant
base_kernel.base_kernel.lengthscale = length_scale

base_kernel.base_kernel.output_scale = output_scale


likelihood.noise_covar.noise = noise_std**2

model = CustomizableGPModel(x, y, mean_module, base_kernel, likelihood)

# Sample from the prior for training data
model.eval()
likelihood.eval()
with torch.no_grad():
    prior_dist = likelihood(model(x))
    y_new = prior_dist.sample()

train_y = y_new[:num_init_train_samples]
test_y = y_new[num_init_train_samples : num_init_train_samples + 3*num_test_samples]
pool_y = y_new[num_init_train_samples + 3*num_test_samples : num_init_train_samples + 3*num_pool_samples + 3*num_test_samples]

Predictor = ConstantValueNetwork(constant_value=0.0, output_size=1)

new_train_x = torch.cat([x[:num_init_train_samples],x[num_init_train_samples+num_test_samples*3+num_pool_samples+1:num_init_train_samples+num_test_samples*3+num_pool_samples*2+2],x[-1:]])
new_train_y = torch.cat([y_new[:num_init_train_samples],y[num_init_train_samples+num_test_samples*3+num_pool_samples+1:num_init_train_samples+num_test_samples*3+num_pool_samples*2+2],y_new[-1:]])

model.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)       ####### CAN ALSO USE TRAINING OVER NLL HERE########


posterior = likelihood(model(x))
posterior_mean = posterior.mean
posterior_var = posterior.variance


# plt.scatter(x,posterior_mean.detach().numpy())
# plt.scatter(x.squeeze(),posterior_mean.detach().numpy()-2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
# plt.scatter(x.squeeze(),posterior_mean.detach().numpy()+2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
# plt.ylim(-2, 2)
# plt.savefig('pg_test_gpr_example_2.jpg')

var_l2_loss_estimator(model, test_x, Predictor, None, None)

query_batch_size = 2
learning_rate_PG = 1e-2
num_episode = 1000
in_dim = input_dim
out_dim = 1
tensor_size = num_pool_samples*3


reciprocal_size_value =  math.log(1.0 / tensor_size)
# Now compute the logarithm of the reciprocal
policy = torch.full([tensor_size], reciprocal_size_value, requires_grad=True)

env = toy_GP_ENV(init_train_x,train_y,test_x,pool_x,pool_y,model,Predictor,query_batch_size)
# policy = MLP_Policy(in_dim,out_dim)
# optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate_PG)
# optimizer = torch.optim.SGD([policy], lr=learning_rate_PG)
optimizer = torch.optim.Adam([policy], lr=learning_rate_PG, weight_decay = 0)
reward_pool = train(env,policy,optimizer,num_episode,pool_x)

# visualize training procedure
data_series = pd.Series(reward_pool)
# rolling_mean = data_series
rolling_mean = data_series.rolling(window=200).mean()
plt.plot(rolling_mean)
plt.savefig('pg_test_gpr.jpg')
