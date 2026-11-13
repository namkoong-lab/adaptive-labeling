# -*- coding: utf-8 -*-
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

import k_subset_sampling
#from nn_feature_weights import NN_feature_weights
from sample_normal import sample_multivariate_normal
from gaussian_process_cholesky_advanced import RBFKernelAdvanced, GaussianProcessCholeskyAdvanced
from variance_ate import var_ate_estimator, ate, var_ate_custom_gp_estimator
from custom_gp_cholesky import GaussianProcessCholesky, RBFKernel

# Define a configuration class for dataset-related parameters
@dataclass
class DatasetConfig:
    def __init__(self, direct_tensors_bool: bool, csv_file_train=None, csv_file_test=None, csv_file_pool=None, y_column=None):
        self.direct_tensors_bool = direct_tensors_bool
        self.csv_file_train = csv_file_train
        self.csv_file_test = csv_file_test
        self.csv_file_pool = csv_file_pool
        self.y_column = y_column      # Assuming same column name across above 3 sets


@dataclass
class ModelConfig:
    access_to_true_pool_y: bool
    hyperparameter_tune: bool
    batch_size_query: int
    temp_k_subset: float
    meta_opt_lr: float
    meta_opt_weight_decay: float


@dataclass
class TrainConfig:
    n_train_iter: int
    n_samples: int
    G_samples: int


@dataclass
class GPConfig:
    length_scale: float
    output_scale: float
    noise_var: float
    parameter_tune_lr: float
    parameter_tune_weight_decay: float
    parameter_tune_nepochs: int
    stabilizing_constant: float
    
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")    

def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, gp_config: GPConfig, direct_tensor_files, Predictor, device, if_print = 0):


    # Predictor here has already been pretrained


    # ------ see how to define a global seed --------- and separate controllable seeds for reproducibility
    #torch.manual_seed(40)



    #if dataset_config.large_dataset:

    #   dataset_train = TabularDataset(device, csv_file=dataset_config.csv_file_train, y_column=dataset_config.y_column)
    #   dataloader_train = DataLoader(dataset_train, batch_size=model_config.batch_size_train, shuffle=True)     # gives batch for training features and labels  (both in float 32)

    #   dataset_test = TabularDataset(device, csv_file=dataset_config.csv_file_test, y_column=dataset_config.y_column)
    #   dataloader_test = DataLoader(dataset_test, batch_size=model_config.batch_size_test, shuffle=False)       # gives batch for test features and label    (both in float 32)

    #   dataset_pool = TabularDataset(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    #   pool_size = len(dataset_pool)
    #   dataloader_pool = DataLoader(dataset_pool, batch_size=pool_size, shuffle=False)       # gives all the pool features and label   (both in float 32) - needed for input in NN_weights

    #   dataset_pool_train = TabularDatasetPool(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    #   dataloader_pool_train = DataLoader(dataset_pool_train, batch_size=model_config.batch_size_train, shuffle=True)       # gives batch of the pool features and label   (both in float 32) - needed for updating the posterior of ENN - as we will do batchwise update


    #else:
    
    if dataset_config.direct_tensors_bool:
        assert direct_tensor_files != None, "direct_tensors_were_not_provided"
        init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx = direct_tensor_files
    
    
    
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
        
    
    
    
    pool_size = pool_x.size(0)




    #input_feature_size = init_train_x.size(1)
    #NN_weights = NN_feature_weights(input_feature_size, model_config.hidden_sizes_weight_NN, 1).to(device)
    #meta_opt = optim.Adam(NN_weights.parameters(), lr=model_config.meta_opt_lr, weight_decay=model_config.meta_opt_weight_decay)


    # Convert the size to a tensor and calculate the reciprocal
    #reciprocal_size_value =  math.log(1.0 / pool_size)
    NN_weights = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device=device)
    #print("1:",NN_weights.is_leaf)
    meta_opt = optim.Adam([NN_weights], lr=model_config.meta_opt_lr, weight_decay=model_config.meta_opt_weight_decay)
    #print("2:",NN_weights.is_leaf)

    SubsetOperatorthis = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, False).to(device)

    #seed for this
    SubsetOperatortestthis = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, True).to(device)

    #if dataset_config.large_dataset:
    #  train_smaller_dataset(init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperator, Predictor, if_print = if_print)
    #  test_smaller_dataset(init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperator, Predictor, if_print = if_print)

    #else:

    if model_config.hyperparameter_tune:
      gp_model = GaussianProcessCholeskyAdvanced(length_scale_init=gp_config.length_scale, variance_init=gp_config.output_scale, noise_var_init=gp_config.noise_var).to(device)
      optimizer = torch.optim.Adam(gp_model.parameters(), lr=gp_model.parameter_tune_lr, weight_decay = gp_model.parameter_tune_weight_decay)

      gp_model.train()  # Set the model to training mode
      for epoch in range(gp_model.parameter_tune_nepochs):
        optimizer.zero_grad()  # Clear previous gradients
        loss = gp_model.nll(init_train_x, init_train_y)  # Compute the loss (NLL)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        if (epoch + 1) % 10 == 0:
            print_model_parameters(gp_model)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    else:
        kernel = RBFKernel(length_scale=gp_config.length_scale, output_scale = gp_config.output_scale).to(device)
        gp_model = GaussianProcessCholesky(kernel=kernel).to(device)






    train_smaller_dataset(gp_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperatorthis, Predictor, pool_sample_idx, if_print = if_print)
    var_ate = test_smaller_dataset(gp_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperatortestthis, Predictor, pool_sample_idx, if_print = if_print)
    
    return var_ate

def train_smaller_dataset(gp_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperatorthis, Predictor, pool_sample_idx, if_print = 0):
  print("NN_weights_in_start:", NN_weights) 
  #print("3:",NN_weights.is_leaf)
  for i in range(train_config.n_train_iter):    # Should we do this multiple times or not
    start_time = time.time()

    meta_opt.zero_grad()
    average_meta_loss = 0.0

    #pool_weights = NN_weights(pool_x)   #pool_weights has shape [pool_size,1]
    #pool_weights_t = pool_weights.t()  #convert pool_weights to shape 
    #soft_k_vector = SubsetOperator(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]

     
    if model_config.access_to_true_pool_y:
        y_gp = torch.cat([init_train_y,pool_y], dim=0)      # [init_train_size(0)+pool_size(0)]
    else:
        w_dumi = torch.ones(init_train_batch_size).to(device)
        mu1, cov1 = gp_model(init_train_x, init_train_y, w_dumi, pool_x, gp_config.stabilizing_constant, gp_config.noise_var)
        cov_final = cov1 +  gp_config.noise_var * torch.eye(pool_x.size(0), device=pool_x.device)
        pool_y_dumi = sample_multivariate_normal(mu1, cov_final, 1)
            #print(pool_y_dumi)
        y_gp = torch.cat([init_train_y,pool_y_dumi], dim=0)

    for g in range(train_config.G_samples):
       

        NN_weights_unsqueezed = NN_weights.unsqueeze(0)       #[1, pool_size]
        soft_k_vector = SubsetOperatorthis(NN_weights_unsqueezed)  #soft_k_vector has shape  [1,pool_size]
        soft_k_vector_squeeze = soft_k_vector.squeeze()  #soft_k_vector_squeeze has shape  [pool_size]
        clipped_soft_k_vector_squeeze = torch.clamp(soft_k_vector_squeeze, min=-float('inf'), max=1.0)

        print(clipped_soft_k_vector_squeeze)
        input_feature_size = init_train_x.size(1)
        init_train_batch_size = init_train_x.size(0)


        

        x_gp = torch.cat([init_train_x,pool_x], dim=0)
        w_train = torch.ones(init_train_batch_size, requires_grad = True).to(device)
        w_gp = torch.cat([w_train,clipped_soft_k_vector_squeeze])



        mu2, cov2 = gp_model(x_gp, y_gp, w_gp, test_x, gp_config.stabilizing_constant, gp_config.noise_var)
        mean_ate, var_ate = var_ate_custom_gp_estimator(mu2, cov2, gp_config.noise_var, test_x, Predictor, device, train_config.n_samples)
        var_ate = var_ate/train_config.G_samples
        var_ate.backward()
        average_meta_loss += var_ate
    meta_opt.step()
    ate_actual = ate(test_x, test_y, Predictor, None)
    #print("4:",NN_weights.is_leaf)

    _, indices = torch.topk(NN_weights, model_config.batch_size_query)
    hard_k_vector = torch.zeros_like(NN_weights)
    hard_k_vector[indices] = 1.0
    y_gp_hard = torch.cat([init_train_y,pool_y], dim=0)
    x_gp_hard = torch.cat([init_train_x,pool_x], dim=0)
    w_train_hard = torch.ones(init_train_batch_size, requires_grad = True).to(device)
    #w_gp = torch.cat([w_train,soft_k_vector_squeeze])
    w_gp_hard = torch.cat([w_train_hard,hard_k_vector])
    mu2_hard, cov2_hard = gp_model(x_gp_hard, y_gp_hard, w_gp_hard, test_x, gp_config.stabilizing_constant, gp_config.noise_var)

    mean_ate_hard, var_ate_hard = var_ate_custom_gp_estimator(mu2_hard, cov2_hard, gp_config.noise_var, test_x, Predictor, device, train_config.n_samples)
    
    #print("5:",NN_weights.is_leaf)
    if pool_sample_idx != None:
        NN_weights_values, NN_weights_indices = torch.topk(NN_weights, model_config.batch_size_query)
        #selected_clusters_from_pool = pool_sample_idx[NN_weights_indices]
        #selected_points_indices = {f"selected_point_{j}_indices": NN_weights_indices[j].item() for j in range(model_config.batch_size_query)}
        #selected_clusters_from_pool_tensor_data = {f"selected_point_{j}_belongs_to_the_cluster": selected_clusters_from_pool[j].item() for j in range(model_config.batch_size_query)}
        #wandb.log({"epoch": i, "var_ate": average_meta_loss.item(), "var_ate_hard":var_ate_hard.item(),"mean_ate": mean_ate.item(), "ate_actual":ate_actual.item(),**selected_points_indices,**selected_clusters_from_pool_tensor_data})
        weights_dict = {f"weight_{j}": NN_weights[j].detach().cpu().item() for j in range(NN_weights.size(0))}
        wandb.log({"epoch": i, "aeverage_var_ate": average_meta_loss.item(), "var_ate_hard":var_ate_hard.item(),"mean_ate": mean_ate.item(), "ate_actual":ate_actual.item(), **weights_dict})
        #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
    else:
        weights_dict = {f"weights/weight_{i}": weight.detach().cpu().item() for i, weight in enumerate(NN_weights)}
        wandb.log({"epoch": i, "var_ate": average_meta_loss.item(), "var_ate_hard":var_ate_hard.item(), "mean_ate": mean_ate.item(), "ate_actual":ate_actual.item(), **weights_dict})
        #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
        
        #wandb.log(weights_dict)
    
    


def test_smaller_dataset(gp_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, model_config, train_config, gp_config, NN_weights, meta_opt, SubsetOperatortestthis, Predictor, pool_sample_idx, if_print = 0):


    _, indices = torch.topk(NN_weights, model_config.batch_size_query)
    hard_k_vector = torch.zeros_like(NN_weights)
    hard_k_vector[indices] = 1.0
    
    #NN_weights_unsqueezed = NN_weights.unsqueeze(0)
    #soft_k_vector = SubsetOperatortestthis(NN_weights_unsqueezed)
    #soft_k_vector = SubsetOperator(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]
    #soft_k_vector_squeeze = soft_k_vector.squeeze()

    #print(soft_k_vector_squeeze)
    input_feature_size = init_train_x.size(1)
    init_train_batch_size = init_train_x.size(0)

    y_gp = torch.cat([init_train_y,pool_y], dim=0)
    x_gp = torch.cat([init_train_x,pool_x], dim=0)
    w_train = torch.ones(init_train_batch_size, requires_grad = True).to(device)
    #w_gp = torch.cat([w_train,soft_k_vector_squeeze])
    w_gp = torch.cat([w_train,hard_k_vector])
    mu2, cov2 = gp_model(x_gp, y_gp, w_gp, test_x, gp_config.stabilizing_constant, gp_config.noise_var)

    mean_ate, var_ate = var_ate_custom_gp_estimator(mu2, cov2, gp_config.noise_var, test_x, Predictor, device, train_config.n_samples)
    #print("var_square_loss:", var_square_loss)


    ate_actual = ate(test_x, test_y, Predictor, None)
    #print("ate_actual:", ate_actual)
    
    
    
    if pool_sample_idx != None:
        NN_weights_values, NN_weights_indices = torch.topk(NN_weights, model_config.batch_size_query)
        #selected_clusters_from_pool = pool_sample_idx[NN_weights_indices]
        #selected_points_indices = {f"val_selected_point_{j}_indices": NN_weights_indices[j].item() for j in range(model_config.batch_size_query)}
        #selected_clusters_from_pool_tensor_data = {f"val_selected_point_{j}_belongs_to_the_cluster": selected_clusters_from_pool[j].item() for j in range(model_config.batch_size_query)}
        #wandb.log({"val_var_ate": var_ate.item(), "val_mean_ate": mean_ate.item(), "val_ate_actual":ate_actual.item(), **selected_points_indices, **selected_clusters_from_pool_tensor_data})
        wandb.log({"val_var_ate": var_ate.item(), "val_mean_ate": mean_ate.item(), "val_ate_actual":ate_actual.item()})
        
    else:
        wandb.log({"val_var_ate": var_ate.item(), "val_mean_ate": mean_ate.item(), "val_ate_actual":ate_actual.item()})
    
    print("NN_weights_in_end:", NN_weights)
    fig2 = plt.figure()
    plt.scatter(init_train_x.cpu(),  init_train_y.cpu(), label='Initial labeled data')
    plt.scatter(test_x.cpu(),  test_y.cpu(), label='Population distribution')

    # Annotate each point in pool_x with its index
    for (i, x, y) in zip(indices.detach().cpu().numpy(), pool_x[indices].cpu().numpy(), pool_y[indices].cpu().numpy()):
        plt.annotate(i, (x, y))

    plt.scatter(pool_x[indices].cpu(), pool_y[indices].cpu(), label='Batch selected')
    plt.legend()
    wandb.log({"env_plot_with_pool_indexes_selected": wandb.Image(fig2)})
    plt.close(fig2)


    
    return var_ate