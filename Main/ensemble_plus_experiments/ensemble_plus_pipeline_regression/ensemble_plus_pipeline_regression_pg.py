# -*- coding: utf-8 -*-
"""GP_pipeline_regression_true_pool_label_true_GP_params.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jonUWpD_3A9GDxI1N4oM5LYdgB0DoMSa
https://github.com/namkoong-lab/adaptive_sampling/blob/c1b7fe65bffeff2f731fa0030ad170456c26d316/src/baselines/RL_scripts/gp_pipeline_regression_modified.py
"""

import argparse
import typing

import torch
import gpytorch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as distributions
import numpy as np
from dataclasses import dataclass
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn as nn
from torch import Tensor
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_


# import k_subset_sampling
#from nn_feature_weights import NN_feature_weights
# from sample_normal import sample_multivariate_normal
# from gaussian_process_cholesky_advanced import RBFKernelAdvanced, GaussianProcessCholeskyAdvanced

from dataloader_enn import TabularDataset, TabularDatasetPool, TabularDatasetCsv, TabularDatasetPoolCsv, BootstrappedSampler
from enn import ensemble_base, ensemble_prior
from variance_l_2_loss_enn import l2_loss, var_l2_loss_estimator
from enn_loss_func import weighted_l2_loss

#from variance_l_2_loss import var_l2_loss_estimator, l2_loss, var_l2_loss_estimator_pg  - - corresponding to gp


#from polyadic_sampler_new import CustomizableGPModel
# from custom_gp_cholesky import GaussianProcessCholesky, RBFKernel

import gymnasium as gym
from gymnasium import spaces

from tqdm import tqdm



# Define a configuration class for dataset-related parameters
@dataclass
class DatasetConfig:
    def __init__(self, direct_tensors_bool: bool, csv_file_train=None, csv_file_test=None, csv_file_pool=None, y_column=None, shuffle=False):
        self.direct_tensors_bool = direct_tensors_bool
        self.csv_file_train = csv_file_train
        self.csv_file_test = csv_file_test
        self.csv_file_pool = csv_file_pool
        self.y_column = y_column      # Assuming same column name across above 3 sets
        self.shuffle = shuffle    # added for the ensemble_plus 




@dataclass
class ModelConfig:
    access_to_true_pool_y: bool
    batch_size_query: int
    temp_k_subset: float
    meta_opt_lr: float
    meta_opt_weight_decay: float
    n_classes: int


# @dataclass
# class ModelConfig:
#     access_to_true_pool_y: bool
#     hyperparameter_tune: bool
#     batch_size_query: int
#     temp_k_subset: float
#     meta_opt_lr: float
#     meta_opt_weight_decay: float



@dataclass
class TrainConfig:
    n_train_iter: int
    n_samples: int       # to calculate the variance
    G_samples: int
    n_iter_noise: int     # not used in regression but used in recall
    batch_size: int

# @dataclass
# class TrainConfig:
#     n_train_iter: int
#     n_samples: int
#     G_samples: int




@dataclass
class ENNConfig:
    basenet_hidden_sizes: list
    exposed_layers: list
    z_dim: int
    learnable_epinet_hiddens: list
    hidden_sizes_prior: list
    seed_base: int
    seed_learnable_epinet: int
    seed_prior_epinet: int
    alpha: float
    n_ENN_iter: int
    ENN_opt_lr: float
    ENN_opt_weight_decay: float
    z_samples: int                         #z_samples in training
    stdev_noise: float

# @dataclass
# class GPConfig:
#     length_scale: float
#     noise_var: float
#     output_scale: float


def plot_visualization(train_x, train_y, step, version = ''):
    if train_x.size(1) == 1: 
    
      fig2 = plt.figure()
      plt.scatter(train_x.to("cpu"),  train_y.to("cpu"), label='Train')

      wandb.log({"Acquired points at step"+str(step)+version: wandb.Image(fig2)})
      plt.close(fig2)

    
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")    

def parameter_regularization_loss(model, initial_parameters, regularization_strength):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        initial_param = initial_parameters[name]
        reg_loss += torch.sum((param) ** 2)
    return reg_loss * regularization_strength

def restore_model(model, saved_state):
    model.load_state_dict(saved_state)

def plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, fnet_loss_list, test_x, test_y, init_train_x, i, device, label_plot=" "):

    if i <=50  or i >= train_config.n_train_iter-2: #only plot first few
        
        prediction_list=torch.empty((0), dtype=torch.float32, device=device)
     
        for z_test in range(enn_config.z_dim):
            #z_test = torch.randn(enn_config.z_dim, device=device)
            prediction = ENN_base(test_x,z_test) + enn_config.alpha * ENN_prior(test_x, z_test) #x is all data
            prediction_list = torch.cat((prediction_list,prediction),1)
        
        posterior_mean = torch.mean(prediction_list, axis = 1)
        posterior_std = torch.std(prediction_list, axis = 1)
    

        fig_fnet_training = plt.figure()
        plt.plot(list(range(len(fnet_loss_list))),fnet_loss_list)
        plt.title('fnet loss within training at training iter '+label_plot + str(i))
        plt.legend()
        wandb.log({'Fnet training loss'+label_plot+ str(i): wandb.Image(fig_fnet_training)})
        plt.close(fig_fnet_training)

        if init_train_x.size(1) == 1:

            fig_fnet_posterior = plt.figure()
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy())
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()-2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()+2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            plt.title('fnet posterior within training at training iter '+label_plot + str(i))
            wandb.log({'Fnet posterior'+label_plot+ str(i): wandb.Image(fig_fnet_posterior)})
            plt.close(fig_fnet_posterior)    

class Ensemble_plus_experiment():
    """
    experiment for training Ensemble_plus (UQ)
    """

    def __init__(self,ENN_base, ENN_prior, enn_config, model_config, train_config, dataset_config, initial_parameters, train_x, train_y, test_x, test_y, Predictor, device):

        self.ENN_base = ENN_base
        self.ENN_prior = ENN_prior
        self.enn_config = enn_config
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model_config = model_config
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.initial_parameters = initial_parameters
        self.Predictor=Predictor
        self.device = device

    def step(self,x,y):
        """ENN training Step"""
        # features,labels = batch.tensors
        new_train_x = torch.cat([self.train_x,x],dim=0)
        new_train_y = torch.cat([self.train_y,y],dim=0)
        
        dataset_train = TabularDataset(x = new_train_x, y = new_train_y)
        dataloader_train = DataLoader(dataset_train, batch_size=self.train_config.batch_size, shuffle=False)

        restore_model(self.ENN_base, self.initial_parameters)
        optimizer_init = optim.Adam(self.ENN_base.parameters(), lr=self.enn_config.ENN_opt_lr, weight_decay=0.0)
        enn_loss_list = []

        weights = []
        for z in range(self.enn_config.z_dim):
            if self.dataset_config.shuffle:
                weights.append(2.0*torch.bernoulli(torch.full((len(dataset_train),), 0.5)).to(self.device))
            else:
                weights.append(torch.ones(len(dataset_train)).to(self.device)) 
        
        for i in range(self.enn_config.n_ENN_iter):
            self.ENN_base.train()
            for (inputs, labels) in dataloader_train:   #check what is dim of inputs, labels, ENN_model(inputs,z)
                aeverage_loss = 0
                optimizer_init.zero_grad()
                for z in range(self.enn_config.z_dim): 
                    #z = torch.randn(enn_config.z_dim, device=device)
                    
                    outputs = self.ENN_base(inputs,z) + self.enn_config.alpha * self.ENN_prior(inputs,z)
                    
                    #print("outputs:", outputs)
                    #print("labels:", labels)
                    #labels = torch.tensor(labels, dtype=torch.long, device=device)

                    loss = weighted_l2_loss(outputs, labels.unsqueeze(dim=1), weights[z])/self.enn_config.z_samples
                    
                    #loss = loss_fn_init(outputs, labels.unsqueeze(dim=1))/enn_config.z_samples
                    reg_loss = parameter_regularization_loss(self.ENN_base, self.initial_parameters, self.enn_config.ENN_opt_weight_decay)/self.enn_config.z_samples
                    loss= loss+reg_loss
                    loss.backward()
                    aeverage_loss += loss
                #clip_grad_norm_(ENN_base.parameters(), max_norm=2.0)
                optimizer_init.step()
                
                enn_loss_list.append(float(aeverage_loss.detach().to('cpu').numpy()))
        
        # prediction_list=torch.empty((0), dtype=torch.float32, device=device)
        # #print("model params 2")
        # #print_model_parameters(ENN_model)

        
        # for z_test in range(enn_config.z_dim):
        #     #z_test = torch.randn(enn_config.z_dim, device=device)
        #     prediction = ENN_model(test_x, z_test) #x is all data
        #     prediction_list = torch.cat((prediction_list,prediction),1)
        
        # posterior_mean = torch.mean(prediction_list, axis = 1)
        # posterior_std = torch.std(prediction_list, axis = 1)
        

        plot_ENN_training_posterior(self.ENN_base, self.ENN_prior, self.train_config, self.enn_config, enn_loss_list, self.test_x, self.test_y, new_train_x, -1, self.device)
        meta_mean, meta_loss = var_l2_loss_estimator(self.ENN_base, self.ENN_prior, self.test_x, self.Predictor, self.device, self.enn_config.z_dim, self.enn_config.alpha, self.enn_config.stdev_noise)
        l_2_loss_actual = l2_loss(self.test_x, self.test_y, self.Predictor, None)
        wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})

        return meta_mean, meta_loss

            #self.model.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)

class toy_Ensemble_plus_ENV(gym.Env):

    #T = 1
    # in_dim = 20
    # out_dim = 1
    # num_epochs = 10 #number of epochs for each batch in MLP experiment

    def __init__(self,train_x,train_y,test_x,test_y,pool_x,pool_y,ENN_base,ENN_prior,model_config, train_config, enn_config, dataset_config, initial_parameters, Predictor, device):

        super().__init__()
        #self.batch_size = batch_size
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.pool_x = pool_x
        self.pool_y = pool_y
        self.ENN_base = ENN_base
        self.ENN_prior = ENN_prior
        self.model_config = model_config
        self.train_config = train_config
        self.enn_config = enn_config
        self.dataset_config = dataset_config
        self.initial_parameters = initial_parameters
        self.Predictor = Predictor
        self.device = device
        self.n_samples = train_config.n_samples
        self.T = 1
        # self.learning_rate = learning_rate
        self.t = 0
        #self.seed_model = seed_model
        # num_dataset = len(dataset)
        self.experiment = Ensemble_plus_experiment(self.ENN_base,self.ENN_prior, self.enn_config, self.model_config, self.train_config, self.dataset_config, self.initial_parameters, self.train_x, self.train_y, self.test_x, self.test_y, self.Predictor, self.device)

        #ENN_base, ENN_prior, enn_config, model_config, dataset_config, initial_parameters, train_x, train_y, test_x, test_y, Predictor, device)
        # self.init_model = copy.deepcopy(self.experiment.model)
        #self.action_space = spaces.MultiBinary(num_dataset) #gym settings currently not needed
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(num_dataset,)) #gym settings currently not needed
        print("INITIALIZED")

    def reset(self, seed=None, options=None):
        return None

    def _get_obs(self):
        return None

    def _get_info(self):
        return None
 
    def step(self, action):
        """environment step"""
        x = self.pool_x[action]
        y = self.pool_y[action]
        mean, loss = self.experiment.step(x,y)
        #observation = self._get_obs() # not needed
        #mean, loss = var_l2_loss_estimator(self.experiment.model, self.test_x, self.Predictor, (self.test_x).device,self.n_samples)
        #print("meanloop:", mean)
        #print("lossloop:", loss)

        terminated = False # check if it should terminate (we currently just have 1 step)
        self.t += 1
        if self.t >= self.T:
            terminated = True

        #truncated = False
        #terminated = True
        #info = self._get_info()

        return mean, loss, terminated

    def render(self):
        pass

    def close(self):
        pass


# def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, gp_config: GPConfig, direct_tensor_files, Predictor, device, if_print = 0):
    
#     if dataset_config.direct_tensors_bool:
#         assert direct_tensor_files != None, "direct_tensors_were_not_provided"
#         init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx = direct_tensor_files
    
#     else: 
#         init_train_data_frame = pd.read_csv(dataset_config.csv_file_train)
#         pool_data_frame = pd.read_csv(dataset_config.csv_file_pool)
#         test_data_frame = pd.read_csv(dataset_config.csv_file_test)
#         init_train_x = torch.tensor(init_train_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
#         init_train_y = torch.tensor(init_train_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
#         pool_x = torch.tensor(pool_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
#         pool_y = torch.tensor(pool_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
#         test_x = torch.tensor(test_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
#         test_y = torch.tensor(test_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
#         pool_sample_idx = None 
#         test_sample_idx = None
    
#     pool_size = pool_x.size(0)

#     # mean_module = gpytorch.means.ConstantMean()
#     # base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#     # likelihood = gpytorch.likelihoods.GaussianLikelihood()


#     # length_scale = gp_config.length_scale
#     # noise_var = gp_config.noise_var
#     # output_scale = gp_config.output_scale

#     # mean_module.constant = 0.0
#     # base_kernel.base_kernel.lengthscale = length_scale
#     # base_kernel.outputscale = output_scale
#     # likelihood.noise_covar.noise = noise_var


#     # gp_model = CustomizableGPModel(init_train_x, init_train_y, mean_module, base_kernel, likelihood).to(device)

#     # # Sample from the prior for training data
#     # gp_model.eval()
#     # likelihood.eval()

#     var_square_loss, policy = policy_gradient_train(gp_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, model_config, train_config, gp_config, Predictor)
    
#     return var_square_loss

def policy_gradient_train(train_x,train_y,test_x,test_y,pool_x,pool_y,ENN_base,ENN_prior,model_config, train_config, enn_config, dataset_config, initial_parameters, Predictor,device):

    pool_size = pool_x.size(0)
    reciprocal_size_value =  math.log(1.0 / pool_size)
    policy = torch.full([pool_size], reciprocal_size_value, dtype=torch.double, requires_grad=True, device=device)

    batch_size_query = model_config.batch_size_query

    env = toy_Ensemble_plus_ENV(train_x,train_y,test_x,test_y,pool_x,pool_y,ENN_base,ENN_prior,model_config, train_config, enn_config, dataset_config, initial_parameters, Predictor, device=device)

    #env = toy_GP_ENV(init_train_x,init_train_y,test_x,pool_x,pool_y,gp_model,model_config, train_config, gp_config,Predictor,batch_size_query)

    optimizer = torch.optim.Adam([policy], lr=model_config.meta_opt_lr, weight_decay = model_config.meta_opt_weight_decay)
    
    loss_pool = []

    steps = 0

    for episode in tqdm(range(train_config.n_train_iter)):
        #state = env.reset() # reset env, state currenly not needed
        #env.render()

        for t in range(1):
            w = policy.squeeze()
            prob = F.softmax(w, dim=0)
            print("w:",w)
            print("prob:",prob)   
            
            loss_temp = []
            mean_all = []
            for j in range(train_config.G_samples):
                batch_ind = torch.multinomial(prob, batch_size_query, replacement=False)
                log_pr = (torch.log(prob[batch_ind])).sum()
                for i in range(batch_size_query):
                    log_pr = log_pr- torch.log(1 - prob[batch_ind[:i]].sum())
                action = batch_ind

                mean, loss, done = env.step(action) # env step, uq update
                #print("log_pr:", log_pr)
                #print("loss:", loss)
                loss = loss.detach()
                loss = log_pr*loss
                loss_temp.append(loss)
                mean_all.append(mean)
                env.reset()

            avg_loss = torch.stack(loss_temp).mean()
            mean_square_loss = torch.stack(mean_all).mean()

            optimizer.zero_grad()
            avg_loss.backward()
            clip_grad_norm_([policy], max_norm=10.0)
            optimizer.step()
            #print("policy:", policy)





            ######################################################################


            x_combined = torch.cat([train_x, pool_x], dim=0)
            y_combined = torch.cat([train_y, pool_y], dim=0)
            #dataset_train_and_pool = TabularDatasetPool(x=x_combined, y=y_combined)
            #dataloader_train_and_pool = DataLoader(dataset_train_and_pool, batch_size=train_config.batch_size, shuffle=False)

            dataset_train_and_pool_hard = TabularDatasetPool(x=x_combined, y=y_combined)
            dataloader_train_and_pool_hard = DataLoader(dataset_train_and_pool_hard, batch_size=train_config.batch_size, shuffle=False)


            restore_model(ENN_base, initial_parameters)
            _, indices = torch.topk(policy, model_config.batch_size_query)
            plot_visualization(pool_x[indices], pool_y[indices], i, version = 'pool')
            hard_k_vector = torch.zeros_like(policy)
            hard_k_vector[indices] = 1.0
            init_train_size = train_x.size(0)
            w_train = torch.ones(init_train_size, device=device)
            w_enn = torch.cat([w_train,hard_k_vector])

            optimizer_init = optim.Adam(ENN_base.parameters(), lr=enn_config.ENN_opt_lr, weight_decay=0.0)
            enn_loss_list = []
            for abcd in range(enn_config.n_ENN_iter):
                ENN_base.train()
                for (idx_batch, inputs, labels) in dataloader_train_and_pool_hard:   #check what is dim of inputs, labels, ENN_model(inputs,z)
                    aeverage_loss = 0
                    optimizer_init.zero_grad()
                    for z in range(enn_config.z_dim): 
                        #z = torch.randn(enn_config.z_dim, device=device)
                        
                        outputs = ENN_base(inputs,z) + enn_config.alpha * ENN_prior(inputs,z)
                        #print("idx_batch:", idx_batch.detach().to("cpu"))
                        
                        #print("outputs:", outputs)
                        #print("labels:", labels)
                        #labels = torch.tensor(labels, dtype=torch.long, device=device)
                        weights_batch = w_enn[idx_batch]
                        #print("weights_batch:", weights_batch.detach().to("cpu"))
                        #print("w_enn:", w_enn.detach().to("cpu"))

                        loss = weighted_l2_loss(outputs, labels.unsqueeze(dim=1), weights_batch)/enn_config.z_samples
                        
                        #loss = loss_fn_init(outputs, labels.unsqueeze(dim=1))/enn_config.z_samples
                        reg_loss = parameter_regularization_loss(ENN_base, initial_parameters, enn_config.ENN_opt_weight_decay)/enn_config.z_samples
                        loss= loss+reg_loss
                        loss.backward()
                        aeverage_loss += loss
                    clip_grad_norm_(ENN_base.parameters(), max_norm=2.0)
                    optimizer_init.step()
                    
                    enn_loss_list.append(float(aeverage_loss.detach().to('cpu').numpy()))
            

            plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, train_x, 0, device, label_plot="hard"+str(i))
            hard_meta_mean, hard_meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
            hard_l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
            wandb.log({"var_square_loss_hard": hard_meta_loss.item(), "mean_square_loss_hard": hard_meta_mean.item(), "l_2_loss_actual_hard": hard_l_2_loss_actual.item()})

            restore_model(ENN_base, initial_parameters)



            # # chane the y_enn here and dataloaders
            #Train the copy of ENN_model -> ENN_model_new here so that weights at the start of the loop are same
            #var_square_loss_hard = var_l2_loss_estimator(ENN_model_new, test_x, Predictor, device, None)
            #hard_loss_mean, hard_var_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
            weights_dict = {f"weight_{a}": policy[a].detach().cpu().item() for a in range(policy.size(0))}
            wandb.log({"epoch": episode, "aeverage_var_square_loss": avg_loss.item(), "mean_square_loss": mean_square_loss.item(), **weights_dict})
                
            # if pool_sample_idx != None:
            #     #NN_weights_values, NN_weights_indices = torch.topk(NN_weights, model_config.batch_size_query)
            #     #selected_clusters_from_pool = pool_sample_idx[NN_weights_indices]
            #     #selected_points_indices = {f"selected_point_{j}_indices": NN_weights_indices[j].item() for j in range(model_config.batch_size_query)}
            #     #selected_clusters_from_pool_tensor_data = {f"selected_point_{j}_belongs_to_the_cluster": selected_clusters_from_pool[j].item() for j in range(model_config.batch_size_query)}
            #     #wandb.log({"epoch": i, "var_square_loss": average_meta_loss.item(), "var_square_loss_hard":var_square_loss_hard.item(),"mean_square_loss": mean_square_loss.item(), "l_2_loss_actual":l_2_loss_actual.item(),**selected_points_indices,**selected_clusters_from_pool_tensor_data})
            #     #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
            # else:
            #     weights_dict = {f"weights/weight_{a}": weight.detach().cpu().item() for a, weight in enumerate(NN_weights)}
            #     wandb.log({"epoch": i,  "time_taken_per_epoch":intermediate_time_3-start_time, "var_square_loss": aeverage_meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual":l_2_loss_actual.item(), **weights_dict})
            #     #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
                
            #     #wandb.log(weights_dict)


            ##################################################



                          
            env.render()

            loss_pool.append(avg_loss.detach().cpu().numpy())

            steps += 1

            if done:
                break

    data_series = pd.Series(loss_pool)
    # rolling_mean = data_series
    rolling_mean = data_series.rolling(window=200).mean()
    plt.plot(rolling_mean)
    plt.savefig('pg_test_gpr.jpg')

    return loss_pool[-1], policy





def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, enn_config: ENNConfig, direct_tensor_files, Predictor, device, seed_training, if_print = 0):
    
    if dataset_config.direct_tensors_bool:
        assert direct_tensor_files != None, "direct_tensors_were_not_provided"
        init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx = direct_tensor_files
        #init_train_x, init_train_y, pool_x, pool_y, test_x, test_y = init_train_x.double(), init_train_y.double(), pool_x.double(), pool_y.double(), test_x.double(), test_y.double()
    
    else: 
        init_train_data_frame = pd.read_csv(dataset_config.csv_file_train)
        pool_data_frame = pd.read_csv(dataset_config.csv_file_pool)
        test_data_frame = pd.read_csv(dataset_config.csv_file_test)
        init_train_x = torch.tensor(init_train_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
        init_train_y = torch.tensor(init_train_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
        pool_x = torch.tensor(pool_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
        pool_y = torch.tensor(pool_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
        test_x = torch.tensor(test_data_frame.drop(dataset_config.y_column, axis=1).values, dtype=torch.float32).to(device)
        test_y = torch.tensor(test_data_frame[dataset_config.y_column].values, dtype=torch.float32).to(device)
        pool_sample_idx = None 
        test_sample_idx = None


    dataset_train = TabularDataset(x = init_train_x, y = init_train_y)
    dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)    
    
    pool_size = pool_x.size(0)
    sample, label = dataset_train[0]
    input_feature_size = sample.shape[0]

    ENN_base = ensemble_base(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.z_dim, enn_config.seed_base).to(device)
    ENN_prior = ensemble_prior(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.z_dim, enn_config.seed_prior_epinet).to(device)

    # mean_module = gpytorch.means.ConstantMean()
    # base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()


    # length_scale = gp_config.length_scale
    # noise_var = gp_config.noise_var
    # output_scale = gp_config.output_scale

    # mean_module.constant = 0.0
    # base_kernel.base_kernel.lengthscale = length_scale
    # base_kernel.outputscale = output_scale
    # likelihood.noise_covar.noise = noise_var


    # gp_model = CustomizableGPModel(init_train_x, init_train_y, mean_module, base_kernel, likelihood).to(device)
    # gp_model = gp_model.double()

    # # Sample from the prior for training data
    # gp_model.eval()
    # likelihood.eval()

    initial_parameters = {name: param.clone().detach() for name, param in ENN_base.named_parameters()}

    prediction_list=torch.empty((0), dtype=torch.float32, device=device)
     
    for z_test in range(enn_config.z_dim):
        #z_test = torch.randn(enn_config.z_dim, device=device)
        prediction = ENN_base(test_x, z_test) + enn_config.alpha * ENN_prior(test_x, z_test) #x is all data
        prediction_list = torch.cat((prediction_list,prediction),1)
      
    posterior_mean = torch.mean(prediction_list, axis = 1)
    posterior_std = torch.std(prediction_list, axis = 1)
    


    meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
    l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
    wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})

    
    if init_train_x.size(1) == 1:
        fig_enn_posterior = plt.figure()
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy())
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()-2*posterior_std.detach().cpu().numpy(),alpha=0.2)
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()+2*posterior_std.detach().cpu().numpy(),alpha=0.2)
        wandb.log({'ENN initial posterior before training': wandb.Image(fig_enn_posterior)})
        plt.close(fig_enn_posterior)



    # Need to do this because ENN_model itself has some seeds and we need to set the seed for the whole training process here
    torch.manual_seed(seed_training)
    np.random.seed(seed_training)
    if device=="cuda":
        torch.cuda.manual_seed(seed_training) # Sets the seed for the current GPU
        torch.cuda.manual_seed_all(seed_training) # Sets the seed for all GPUs

    weights = []
    for z in range(enn_config.z_dim):
        if dataset_config.shuffle:
            weights.append(2.0*torch.bernoulli(torch.full((len(dataset_train),), 0.5)).to(device))
        else:
            weights.append(torch.ones(len(dataset_train)).to(device))    

    
    optimizer_init = optim.Adam(ENN_base.parameters(), lr=enn_config.ENN_opt_lr, weight_decay=0.0)
    enn_loss_list = []
    for i in range(enn_config.n_ENN_iter):
        ENN_base.train()
        for (inputs, labels) in dataloader_train:   #check what is dim of inputs, labels, ENN_model(inputs,z)
            aeverage_loss = 0
            optimizer_init.zero_grad()
            for z in range(enn_config.z_dim): 
                #z = torch.randn(enn_config.z_dim, device=device)
                
                outputs = ENN_base(inputs,z) + enn_config.alpha * ENN_prior(inputs,z)
                
                #print("outputs:", outputs)
                #print("labels:", labels)
                #labels = torch.tensor(labels, dtype=torch.long, device=device)

                loss = weighted_l2_loss(outputs, labels.unsqueeze(dim=1), weights[z])/enn_config.z_samples
                
                #loss = loss_fn_init(outputs, labels.unsqueeze(dim=1))/enn_config.z_samples
                reg_loss = parameter_regularization_loss(ENN_base, initial_parameters, enn_config.ENN_opt_weight_decay)/enn_config.z_samples
                loss= loss+reg_loss
                loss.backward()
                aeverage_loss += loss
            #clip_grad_norm_(ENN_base.parameters(), max_norm=2.0)
            optimizer_init.step()
            
            enn_loss_list.append(float(aeverage_loss.detach().to('cpu').numpy()))
     
    plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, init_train_x, -1, device)
    meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
    l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
    wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})

    
  


    #train(ENN_base, ENN_prior, initial_parameters, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatorthis, Predictor, pool_sample_idx, if_print = if_print)
    #var_square_loss = test(ENN_base, ENN_prior, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatortestthis, Predictor, pool_sample_idx, if_print = if_print)
    
    #return var_square_loss


    var_square_loss, policy = policy_gradient_train(init_train_x,init_train_y,test_x,test_y,pool_x,pool_y,ENN_base,ENN_prior,model_config, train_config, enn_config, dataset_config, initial_parameters, Predictor, device)
    return var_square_loss, policy