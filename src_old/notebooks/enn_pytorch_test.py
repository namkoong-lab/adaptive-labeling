#!/user/bw2762/.conda/envs/testbed_2/bin/python

import torch
import torch.nn as nn
import torch.optim as optim

from time import time

#METHODOLOGY 1
# An implementation that combines basenet, learnable epinet, prior epinet

# shape of x in [batch_size,x_dim], z is [z_dim]
# assuming input is always included and output is never included
# hidden layers and exposed layers same number of entries

import neural_testbed
from neural_testbed.agents import factories as agent_factories
from neural_testbed.agents.factories.sweeps import testbed_2d as agent_sweeps
from neural_testbed import base
from neural_testbed import generative
from neural_testbed import leaderboard

import jax
import chex

start = time()

class basenet_with_learnable_epinet_and_ensemble_prior(nn.Module):
    def __init__(self, input_size, basenet_hidden_sizes, n_classes, exposed_layers, z_dim, learnable_epinet_hiddens, hidden_sizes_prior, seed_base, seed_learnable_epinet, seed_prior_epinet, alpha):
        super(basenet_with_learnable_epinet_and_ensemble_prior, self).__init__()


        self.z_dim = z_dim
        self.n_classes = n_classes
        self.num_ensemble = z_dim
        self.alpha = alpha


        # Create a list of all sizes (input + hidden + output)
        basenet_all_sizes = [input_size] + basenet_hidden_sizes + [n_classes]
        # print("basenet_all_sizes:", basenet_all_sizes)
        self.basenet_all_sizes = basenet_all_sizes
        exposed_layers = [True]+exposed_layers+[False]     # assuming input is always included and output is never included
        # print("exposed_layers:",exposed_layers)
        self.exposed_layers = exposed_layers

        torch.manual_seed(seed_base)
        # Dynamically create layers
        self.basenet_layers = nn.ModuleList()
        for i in range(len(basenet_all_sizes) - 1):
            self.basenet_layers.append(nn.Linear(basenet_all_sizes[i], basenet_all_sizes[i + 1]))
        # print("basenet_layers:", self.basenet_layers)

        sum_input_base_epi = sum(basenet_all_size for basenet_all_size, exposed_layer in zip(basenet_all_sizes, exposed_layers) if exposed_layer)
        # print("sum_input_base_epi:", sum_input_base_epi)
        learnable_epinet_all_sizes = [sum_input_base_epi+z_dim]    + learnable_epinet_hiddens + [n_classes*z_dim]
        # print("learnable_epinet_all_sizes:", learnable_epinet_all_sizes)
        self.learnable_epinet_all_sizes = learnable_epinet_all_sizes

        torch.manual_seed(seed_learnable_epinet)
        self.learnable_epinet_layers = nn.ModuleList()
        for j in range(len(learnable_epinet_all_sizes) - 1):
            self.learnable_epinet_layers.append(nn.Linear(learnable_epinet_all_sizes[j], learnable_epinet_all_sizes[j + 1]))
        # print("learnable_epinet_layers:", self.learnable_epinet_layers)



        torch.manual_seed(seed_prior_epinet)
        self.ensemble = nn.ModuleList()
        for _ in range(self.num_ensemble):
            layers = []
            all_sizes_prior = [sum_input_base_epi] + hidden_sizes_prior + [n_classes]
            for i in range(len(all_sizes_prior) - 1):
                layer = nn.Linear(all_sizes_prior[i], all_sizes_prior[i + 1])


                # Initialize weights and biases here
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

                layers.append(layer)
                if i < len(all_sizes_prior) - 2:
                    layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            # Freeze the parameters of this MLP
            for param in mlp.parameters():
                param.requires_grad = False

            self.ensemble.append(mlp)

        # print("ensemble:", self.ensemble)





    def forward(self, x, z):
        hidden_outputs = []
        #concatenate_hidden = x   #assuming x is always input
        # print("x:", x)

        for i, (basenet_layer, flag) in enumerate(zip(self.basenet_layers, self.exposed_layers)):
            if flag:
                hidden_outputs.append(x)
                # print("hidden_outputs:", hidden_outputs)

            x = basenet_layer(x)
            # print("x:", x)
            if i < len(self.basenet_layers) - 1:  # Apply activation function except for the output layer
                x = torch.relu(x)
                # print("x:", x)

            #if i>0 and flag:
                #concatenate_hidden = torch.cat(x,concatenate_hidden, dim=1)

        concatenate_hidden = torch.cat(hidden_outputs, dim=1)
        # print("concatenate_hidden:", concatenate_hidden)
        detached_concatenate_hidden = concatenate_hidden.detach()                    ###-------NOT SURE IF BACKPROP WILL WORK PROPERLY THROUGH THIS
        # print("detached_concatenate_hidden:", detached_concatenate_hidden)
        detached_concatenate_hidden_to_prior = concatenate_hidden.detach()
        # print("detached_concatenate_hidden_to_prior :", detached_concatenate_hidden_to_prior)    ###-------NOT SURE IF BACKPROP WILL WORK PROPERLY THROUGH THIS - should we clone and detach


        z_repeated = z.unsqueeze(0).repeat(detached_concatenate_hidden.size(0), 1)
        # print("z_repeated:",z_repeated)
        combined_output = torch.cat([detached_concatenate_hidden,z_repeated], dim=1)
        # print("combined_output:", combined_output)



        for j, learnable_epinet_layer in enumerate(self.learnable_epinet_layers):
            combined_output = learnable_epinet_layer(combined_output)
            # print("combined_output:", combined_output)
            if j < len(self.learnable_epinet_layers) - 1:  # Apply activation function except for the output layer
                combined_output = torch.relu(combined_output)
                # print("combined_output:", combined_output)

        # print("intermediary_check, x:", x)
        # print("intermediary_check, concatenate_hidden:", concatenate_hidden)
        # print("intermediary_check, detached_concatenate_hidden:", detached_concatenate_hidden)
        # print("intermediary_check, detached_concatenate_hidden_to_prior:", detached_concatenate_hidden_to_prior)


        # print("detached_concatenate_hidden_to_prior:", detached_concatenate_hidden_to_prior)
        # reshaped_output = combined_output_learnable.view(inputs.shape[0], self.num_classes, self.z_dim)
        reshaped_epinet_output = torch.reshape(combined_output, (combined_output.shape[0], self.n_classes, self.z_dim))
        # print("reshaped_epinet_output:",reshaped_epinet_output)
        epinet_output = torch.matmul(reshaped_epinet_output, z)
        # print("epinet_output:", epinet_output)

        outputs_prior = [mlp(detached_concatenate_hidden_to_prior) for mlp in self.ensemble]
        # print("outputs_prior:", outputs_prior)
        outputs_prior_tensor = torch.stack(outputs_prior, dim=0)
        # print("outputs_prior_tensor:", outputs_prior_tensor)
        prior_output = torch.einsum('nbo,n->bo', outputs_prior_tensor, z)
        # print("prior_output:", prior_output)
        final_output =  x + epinet_output + self.alpha* prior_output
        # print("final_output:", final_output)



        return final_output




#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)


# input_size = 2
# basenet_hidden_sizes = [5,5]
# n_classes = 2
# exposed_layers = [False, True]
# z_dim = 3
# learnable_epinet_hiddens = [8,8]
# hidden_sizes_prior = [2,2]
# seed_base = 2
# seed_learnable_epinet = 1
# seed_prior_epinet = 0
# alpha = 0.1

input_size = 2
basenet_hidden_sizes = [50,50]
n_classes = 2
exposed_layers = [False, True]
z_dim = 8
learnable_epinet_hiddens = [15,15]
hidden_sizes_prior = [5,5]
seed_base = 2
seed_learnable_epinet = 1
seed_prior_epinet = 0
alpha = 0.1

model = basenet_with_learnable_epinet_and_ensemble_prior(input_size, basenet_hidden_sizes, n_classes, exposed_layers, z_dim, learnable_epinet_hiddens, hidden_sizes_prior, seed_base, seed_learnable_epinet, seed_prior_epinet, alpha)


# x = torch.randn(5,3)
# z=torch.randn(3)
# model(x,z)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# for name, param in model.named_parameters():
#         print(name, param.data)

# print(len(problem.train_data.x[0]))

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch import Tensor
import haiku as hk

import numpy as np

def experiment(model,problem,seed):
    data = problem.train_data

    # Make a dataset
    num_batches = 100
    batch_size = 10
    x_train = np.array(data.x)
    y_train = np.array(data.y)
    dataset = TensorDataset(Tensor(x_train), Tensor(y_train.flatten()).to(torch.int64))
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_batches*batch_size)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    dataiter = iter(trainloader)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    rngs = hk.PRNGSequence(seed)


    i = 0
    j = 0
    # for epoch in range(num_batches):
    for epoch in range(100):
        model.train()
        i+=1
        # print('epoch:',i)
        for (inputs, labels) in trainloader: 
            j+=1
            # print('steps:',j)
            key = next(rngs)
            z = jax.random.normal(key, shape=[8])
            z = np.array(z)
            z = torch.tensor(z)
            optimizer.zero_grad()
            outputs = model(inputs,z)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        j = 0

    def sampler(x,key):
        start = time()
        # print("type of x:",type(x))
        x = np.array(x)
        # x = jax.numpy.array(x)
        # print("first:",time()-start)
        x = torch.from_numpy(x).float()
        # print("secind:",time()-start)
        z = jax.random.normal(key, shape=[8])
        # print("third:",time()-start)
        z = np.array(z)
        # z = jax.numpy.array(z)
        # print("forth:",time()-start)
        z = torch.tensor(z)
        # print("fiveth:",time()-start)

        # z = torch.randn(shape=[3])
        out = model(x,z)
        # print("six:",time()-start)
        result = out.detach().numpy()
        # print("seven:",time()-start)
        return result
    
    return sampler




id_list = ['classification_2d/289','classification_2d/319','classification_2d/348','classification_2d/409']

for id in id_list:
    start=time()

    problem = leaderboard.problem_from_id(id)
    enn_sampler = experiment(model,problem,seed=0)
    quality = problem.evaluate_quality(enn_sampler)
    print('-------')
    print(id)
    print(leaderboard.SETTINGS[id])
    print(quality)

    # print('time:',time()-start)



# print('time:',time()-start)

# print(problem.evaluate_quality(enn_sampler))
# plots = generative.sanity_plots(problem, enn_sampler)
# p = plots['more_enn']
# _ = p.draw()
# p.save('test.jpg')