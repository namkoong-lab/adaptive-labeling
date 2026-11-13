from typing import Callable, NamedTuple
import numpy as np
import pandas as pd
import torch
import dataclasses
import gpytorch
import argparse
import json

import warnings
warnings.filterwarnings('ignore')


import torch
import gpytorch
import torchopt
#import higher
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
from torch.nn.utils import clip_grad_norm_


import enn_pipeline_regression_1a as enn_pipeline_regression
import polyadic_sampler_alternate_fixed_anchors_5 as polyadic_sampler
from constant_network import ConstantValueNetwork
import wandb
from matplotlib import pyplot as plt
from enn import ensemble_base, ensemble_prior
from dataloader_enn import TabularDataset, TabularDatasetPool, TabularDatasetCsv, TabularDatasetPoolCsv, BootstrappedSampler



from variance_l_2_loss_enn import l2_loss, var_l2_loss_estimator
from enn_loss_func import weighted_l2_loss

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

def posterior_visualization(model,x,step):
    if x.size(1) == 1:

        fig2 = plt.figure()
        posterior = model.likelihood(model(x))
        posterior_mean = posterior.mean
        posterior_std = torch.sqrt(posterior.variance)
        plt.scatter(x,posterior_mean.detach().numpy())
        plt.scatter(x.squeeze(),posterior_mean.detach().numpy()-2*posterior_std.detach().numpy(),alpha=0.2)
        plt.scatter(x.squeeze(),posterior_mean.detach().numpy()+2*posterior_std.detach().numpy(),alpha=0.2)
        plt.title("Posterior at step "+str(step))
        wandb.log({"Posterior at step"+str(step): wandb.Image(fig2)})
        plt.close(fig2)


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

     


def standardize_tensors(*tensors):
    # Concatenate tensors along the first dimension
    combined_tensor = torch.cat(tensors, dim=0)
    
    # Calculate mean and standard deviation
    mean = combined_tensor.mean(dim=0)
    std = combined_tensor.std(dim=0)
    
    # Standardize each tensor
    standardized_tensors = [(tensor - mean) / torch.where(std == 0, 1, std) for tensor in tensors]
    
    return standardized_tensors


def main_run_func():
    with wandb.init(project=PROJECT_NAME, entity=ENTITY) as run:
        config = wandb.config
        #config_dict = wandb.config.as_dict()
        #parameters_to_hash, hash_value = hash_configuration(config_dict)
        # print(type(config_dict))
        #print('config_dict:', config_dict)
        #print('parameters_to_hash:', parameters_to_hash)
        #print('hash_value:', hash_value)
        #print('config:', config)

        # Load the hyperparameters from WandB config
        no_train_points = config.no_train_points 
        no_test_points = config.no_test_points 
        no_pool_points = config.no_pool_points
        dataset_model_name = config.dataset_model_name   #"GP" or "blr"
        no_anchor_points = config.no_anchor_points 
        input_dim = config.input_dim
        stdev_scale = config.stdev_scale
        stdev_pool_scale = config.stdev_pool_scale
        scaling_factor = config.scaling_factor     # None or float
        scale_by_input_dim = config.scale_by_input_dim
        gp_model_dataset_generation = config.gp_model_dataset_generation   # should be "use_default" or "specify_own"
        #model = config.model
        stdev_blr_w = config.stdev_blr_w
        stdev_blr_noise = config.stdev_blr_noise
        logits =  config.logits
        if_logits = config.if_logits     #true or false
        if_logits_only_pool = config.if_logits_only_pool    #true or false
        plot_folder = config.plot_folder    #none or string
        
        
        direct_tensors_bool = config.direct_tensors_bool  #true or false
        csv_file_train = config.csv_file_train
        csv_file_test = config.csv_file_test
        csv_file_pool = config.csv_file_pool
        y_column = config.y_column
        shuffle = config.shuffle



        access_to_true_pool_y = config.access_to_true_pool_y    #true or false
        batch_size_query = config.batch_size_query
        temp_k_subset = config.temp_k_subset
        meta_opt_lr = config.meta_opt_lr
        meta_opt_weight_decay = config.meta_opt_weight_decay
        n_classes = config.n_classes


        n_train_iter = config.n_train_iter
        n_samples = config.n_samples     #n_samples in variance calculation
        G_samples = config.G_samples     #G_samples in gradient average caluclation
        n_iter_noise = config.n_iter_noise
        batch_size = config.batch_size


        basenet_hidden_sizes = config.basenet_hidden_sizes
        exposed_layers = config.exposed_layers
        z_dim = config.z_dim
        learnable_epinet_hiddens = config.learnable_epinet_hiddens
        hidden_sizes_prior = config.hidden_sizes_prior
        seed_base = config.seed_base
        seed_learnable_epinet = config.seed_learnable_epinet
        seed_prior_epinet = config.seed_prior_epinet
        alpha = config.alpha
        n_ENN_iter = config.n_ENN_iter
        ENN_opt_lr = config.ENN_opt_lr
        ENN_opt_weight_decay = config.ENN_opt_weight_decay
        z_samples = config.z_samples
        stdev_noise = config.stdev_noise

        dataset_mean_constant =  config.dataset_mean_constant 
        dataset_length_scale =  config.dataset_length_scale
        dataset_output_scale =  config.dataset_output_scale
        dataset_noise_std  =  config.dataset_noise_std
       
        seed_dataset = config.seed_dataset 
        seed_training = config.seed_training
        device_index = config.device_index
        algo = config.algo         
        
        
        #print('load:', load)
        #print('load_project_name:', load_project_name)
        #print('load_artifact_name:', load_artifact_name)
        #print('save:', save)
        #print('save_project_name:', save_project_name)
        #print('save_artifact_name:', save_artifact_name)
        if torch.cuda.is_available():
           # Set CUDA device
           torch.cuda.set_device(device_index)  # GPU 0, change accordingly
           #print("Using GPU:", torch.cuda.get_device_name(1))
           #device = torch.cuda.get_device_name(1)
           device = torch.device("cuda")
        else:
           device = torch.device("cpu")



        torch.manual_seed(seed_dataset)
        np.random.seed(seed_dataset)
        if device=="cuda":
            torch.cuda.manual_seed(seed_dataset) # Sets the seed for the current GPU
            torch.cuda.manual_seed_all(seed_dataset) # Sets the seed for all GPUs
        
        
        
        if direct_tensors_bool:
            if dataset_model_name == "blr":
                polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(no_train_points = no_train_points,no_test_points = no_test_points,no_pool_points=no_pool_points,model_name = dataset_model_name,no_anchor_points = no_anchor_points, input_dim = input_dim, stdev_scale=stdev_scale, stdev_pool_scale= stdev_pool_scale, scaling_factor = scaling_factor, scale_by_input_dim=scale_by_input_dim,model = None, stdev_blr_w = stdev_blr_w,stdev_blr_noise = stdev_blr_noise,logits = logits,if_logits = if_logits,if_logits_only_pool = if_logits_only_pool,plot_folder=plot_folder)
                train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)
            elif dataset_model_name == "GP":
                if gp_model_dataset_generation=="use_default":
                    polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(no_train_points = no_train_points,no_test_points = no_test_points,no_pool_points=no_pool_points,model_name = dataset_model_name,no_anchor_points = no_anchor_points, input_dim = input_dim, stdev_scale=stdev_scale,stdev_pool_scale= stdev_pool_scale,scaling_factor = scaling_factor,scale_by_input_dim=scale_by_input_dim,model = None, stdev_blr_w = stdev_blr_w,stdev_blr_noise = stdev_blr_noise,logits = logits,if_logits = if_logits,if_logits_only_pool = if_logits_only_pool,plot_folder=plot_folder)
                    train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)
            
                else:
                    dataset_mean_module = gpytorch.means.ConstantMean()
                    dataset_base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())    #base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))      # use this if you want to set the different length parameter for each dimension of the kernel
                    dataset_likelihood = gpytorch.likelihoods.GaussianLikelihood()


                    dataset_mean_module.constant = torch.tensor([dataset_mean_constant])
                    dataset_base_kernel.base_kernel.lengthscale = dataset_length_scale
                    dataset_base_kernel.outputscale = dataset_output_scale
                    dataset_likelihood.noise_covar.noise = dataset_noise_std**2
                    
                    points_initial = 10
                    dumi_train_x = torch.randn(points_initial, input_dim)
                    dumi_train_y = torch.zeros(points_initial)
                    dataset_model = polyadic_sampler.CustomizableGPModel(dumi_train_x, dumi_train_y, dataset_mean_module, dataset_base_kernel, dataset_likelihood)
                    polyadic_sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(no_train_points = no_train_points,no_test_points = no_test_points,no_pool_points=no_pool_points,model_name = dataset_model_name,no_anchor_points = no_anchor_points, input_dim = input_dim, stdev_scale=stdev_scale,stdev_pool_scale= stdev_pool_scale,scaling_factor = scaling_factor,scale_by_input_dim=scale_by_input_dim,model = dataset_model, stdev_blr_w = stdev_blr_w,stdev_blr_noise = stdev_blr_noise,logits = logits,if_logits = if_logits,if_logits_only_pool = if_logits_only_pool,plot_folder=plot_folder)
                    train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx = polyadic_sampler.set_data_parameters_and_generate(polyadic_sampler_cfg)
                                
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            pool_x= pool_x.to(device)
            pool_y = pool_y.to(device)
            test_sample_idx = test_sample_idx.to(device)
            pool_sample_idx = pool_sample_idx.to(device)

            #train_x, test_x, pool_x = standardize_tensors(train_x, test_x, pool_x)
            direct_tensor_files = (train_x, train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx)
        else:
            direct_tensor_files = None
            

        
        
        
        
        dataset_config = enn_pipeline_regression.DatasetConfig(direct_tensors_bool, csv_file_train, csv_file_test, csv_file_pool, y_column, shuffle)
        model_config = enn_pipeline_regression.ModelConfig(access_to_true_pool_y = access_to_true_pool_y, batch_size_query = batch_size_query, temp_k_subset = temp_k_subset, meta_opt_lr = meta_opt_lr, meta_opt_weight_decay = meta_opt_weight_decay, n_classes = n_classes)
        train_config = enn_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples, n_iter_noise = n_iter_noise, batch_size = batch_size) #temp_var_recall needs to be added as a new variable here i var recall setting
        enn_config = enn_pipeline_regression.ENNConfig(basenet_hidden_sizes = basenet_hidden_sizes, exposed_layers = exposed_layers, z_dim = z_dim, learnable_epinet_hiddens = learnable_epinet_hiddens, hidden_sizes_prior = hidden_sizes_prior, seed_base = seed_base, seed_learnable_epinet = seed_learnable_epinet, seed_prior_epinet = seed_prior_epinet, alpha = alpha, n_ENN_iter = n_ENN_iter, ENN_opt_lr = ENN_opt_lr, ENN_opt_weight_decay = ENN_opt_weight_decay, z_samples = z_samples, stdev_noise=stdev_noise)

        
        
        Predictor = ConstantValueNetwork(constant_value=0.0, output_size=1).to(device)
        Predictor.eval()
        #pool_size = pool_x.size(0)
        #sample, label = dataset_train[0]
        input_feature_size = pool_x.size(1)

        ENN_base = ensemble_base(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.z_dim, enn_config.seed_base).to(device)
        ENN_prior = ensemble_prior(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.z_dim, enn_config.seed_prior_epinet).to(device)
            #print("model params 1")
        #print_model_parameters(ENN_model)
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
        
        if train_x.size(1) == 1:
            fig_enn_posterior = plt.figure()
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy())
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()-2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()+2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            wandb.log({'ENN initial posterior before training': wandb.Image(fig_enn_posterior)})
            plt.close(fig_enn_posterior)


        dataset_train = TabularDataset(x = train_x, y = train_y)
        dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)

        
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
        
        trained_parameters = {name: param.clone().detach() for name, param in ENN_base.named_parameters()}        
        
        # prediction_list=torch.empty((0), dtype=torch.float32, device=device)
        # #print("model params 2")
        # #print_model_parameters(ENN_model)

        
        # for z_test in range(enn_config.z_dim):
        #     #z_test = torch.randn(enn_config.z_dim, device=device)
        #     prediction = ENN_model(test_x, z_test) #x is all data
        #     prediction_list = torch.cat((prediction_list,prediction),1)
        
        # posterior_mean = torch.mean(prediction_list, axis = 1)
        # posterior_std = torch.std(prediction_list, axis = 1)
        

        plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, train_x, -1, device)
        meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
        l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
        wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})
        restore_model(ENN_base, initial_parameters)



        #plot_visualization(train_x, train_y, -1)
        #posterior_visualization(gp_model_track,test_x,-1)

        for a in range(1):
            #var_square_loss, NN_weights = gp_pipeline_regression_pg.long_horizon_experiment(dataset_cfg, model_cfg, train_cfg, gp_cfg, direct_tensor_files, model_predictor, device, if_print = 1)
            #wandb.log({"val_final_var_square_loss": var_square_loss})

            if algo == "random":
                indices = np.random.choice(range(pool_x.shape[0]), model_config.batch_size_query, replace = False)

                remaining_indices = list(set(list(range(pool_x.shape[0]))) - set(indices)) #needes to be checked

                #add those to training
                train_x = torch.cat((train_x, pool_x[indices, ]), 0)
                train_y = torch.cat((train_y, pool_y[indices ]), 0)

                dataset_train = TabularDataset(x = train_x, y = train_y)
                dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)
                weights = []
                for z in range(enn_config.z_dim):
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
                
                # prediction_list=torch.empty((0), dtype=torch.float32, device=device)
                # #print("model params 2")
                # #print_model_parameters(ENN_model)

                
                # for z_test in range(enn_config.z_dim):
                #     #z_test = torch.randn(enn_config.z_dim, device=device)
                #     prediction = ENN_model(test_x, z_test) #x is all data
                #     prediction_list = torch.cat((prediction_list,prediction),1)
                
                # posterior_mean = torch.mean(prediction_list, axis = 1)
                # posterior_std = torch.std(prediction_list, axis = 1)
                

                plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, train_x, -1, device)
                meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
                l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
                wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})
                restore_model(ENN_base, initial_parameters)


                #remove those points from pool 
                pool_x = pool_x[remaining_indices, ]
                pool_y = pool_y[remaining_indices]
                pool_sample_idx = pool_sample_idx[remaining_indices]

            
            elif algo == "uncertainty":

                restore_model(ENN_base, trained_parameters)
                prediction_list=torch.empty((0), dtype=torch.float32, device=device)
                for z_test in range(enn_config.z_dim):
                    #z_test = torch.randn(enn_config.z_dim, device=device)
                    prediction = ENN_base(pool_x, z_test) + enn_config.alpha * ENN_prior(pool_x, z_test) #x is all data
                    prediction_list = torch.cat((prediction_list,prediction),1)
                
                posterior_mean = torch.mean(prediction_list, axis = 1)
                posterior_std = torch.std(prediction_list, axis = 1)

                print("posterior_std:", posterior_std)

                _, indices = torch.topk(posterior_std, model_config.batch_size_query)

                remaining_indices = list(set(list(range(pool_x.shape[0]))) - set(indices)) #needes to be checked

                #add those to training
                train_x = torch.cat((train_x, pool_x[indices, ]), 0)
                train_y = torch.cat((train_y, pool_y[indices ]), 0)
                restore_model(ENN_base, initial_parameters)



                dataset_train = TabularDataset(x = train_x, y = train_y)
                dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)
                weights = []
                for z in range(enn_config.z_dim):
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
                
                # prediction_list=torch.empty((0), dtype=torch.float32, device=device)
                # #print("model params 2")
                # #print_model_parameters(ENN_model)

                
                # for z_test in range(enn_config.z_dim):
                #     #z_test = torch.randn(enn_config.z_dim, device=device)
                #     prediction = ENN_model(test_x, z_test) #x is all data
                #     prediction_list = torch.cat((prediction_list,prediction),1)
                
                # posterior_mean = torch.mean(prediction_list, axis = 1)
                # posterior_std = torch.std(prediction_list, axis = 1)
                

                plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, train_x, -1, device)
                meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
                l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
                wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})
                restore_model(ENN_base, initial_parameters)


                #remove those points from pool 
                pool_x = pool_x[remaining_indices, ]
                pool_y = pool_y[remaining_indices]
                pool_sample_idx = pool_sample_idx[remaining_indices]



            elif algo == "greedy-batch-uncertainty":
                train_x_internal = train_x
                train_y_dumi_internal = train_y
                train_y_internal = train_y
                pool_x_internal = pool_x
                pool_y_internal = pool_y
                test_x_internal = test_x

                restore_model(ENN_base, trained_parameters)

                with torch.no_grad():
                    
                    random_integer = torch.randint(0, enn_config.z_dim, (1,)).item()
                    pool_y_dumi_internal = (ENN_base(pool_x, random_integer) + enn_config.alpha * ENN_prior(pool_x,random_integer)).squeeze().detach()    # assuming this can be handled by the GPUs otherwise put it in batches



                restore_model(ENN_base, initial_parameters)


                

                for _ in range(model_config.batch_size_query):
                    restore_model(ENN_base, initial_parameters)
                    dataset_train = TabularDataset(x = train_x_internal, y = train_y_dumi_internal)
                    dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)
                    weights = []
                    for z in range(enn_config.z_dim):
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
                    
                    # prediction_list=torch.empty((0), dtype=torch.float32, device=device)
                    # #print("model params 2")
                    # #print_model_parameters(ENN_model)

                    
                    # for z_test in range(enn_config.z_dim):
                    #     #z_test = torch.randn(enn_config.z_dim, device=device)
                    #     prediction = ENN_model(test_x, z_test) #x is all data
                    #     prediction_list = torch.cat((prediction_list,prediction),1)
                    
                    # posterior_mean = torch.mean(prediction_list, axis = 1)
                    # posterior_std = torch.std(prediction_list, axis = 1)

                    prediction_list=torch.empty((0), dtype=torch.float32, device=device)
                    for z_test in range(enn_config.z_dim):
                        #z_test = torch.randn(enn_config.z_dim, device=device)
                        prediction = ENN_base(pool_x, z_test) + enn_config.alpha * ENN_prior(pool_x, z_test) #x is all data
                        prediction_list = torch.cat((prediction_list,prediction),1)
                    
                    posterior_mean = torch.mean(prediction_list, axis = 1)
                    posterior_std = torch.std(prediction_list, axis = 1)
                    _, indices = torch.topk(posterior_std, 1)

                    remaining_indices_internal = list(set(list(range(pool_x_internal.shape[0]))) - set(indices))
                    train_x_internal = torch.cat((train_x_internal, pool_x_internal[indices, ]), 0)
                    train_y_dumi_internal= torch.cat((train_y_internal, pool_y_dumi_internal[indices ]), 0)
                    train_y_internal = torch.cat((train_y_internal, pool_y_internal[indices ]), 0)
                    pool_x_internal = pool_x_internal[remaining_indices_internal, ]
                    pool_y_dumi_internal = pool_y_dumi_internal[remaining_indices_internal]
                    pool_y_internal = pool_y_internal[remaining_indices_internal]

                train_x = train_x_internal
                train_y = train_y_internal
                pool_x = pool_x_internal
                pool_y = pool_y_internal


                restore_model(ENN_base, initial_parameters)
                dataset_train = TabularDataset(x = train_x, y = train_y)
                dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=False)
                weights = []
                for z in range(enn_config.z_dim):
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
                
             
               

                plot_ENN_training_posterior(ENN_base, ENN_prior, train_config, enn_config, enn_loss_list, test_x, test_y, train_x, -1, device)
                meta_mean, meta_loss = var_l2_loss_estimator(ENN_base, ENN_prior, test_x, Predictor, device, enn_config.z_dim, enn_config.alpha, enn_config.stdev_noise)
                l_2_loss_actual = l2_loss(test_x, test_y, Predictor, None)
                wandb.log({"var_square_loss": meta_loss.item(), "mean_square_loss": meta_mean.item(), "l_2_loss_actual": l_2_loss_actual.item()})
                restore_model(ENN_base, initial_parameters)

            
            #plot_visualization(train_x, train_y, a)
            #posterior_visualization(gp_model_track,test_x,a)





        
        #var_square_loss = enn_pipeline_regression.experiment(dataset_config, model_config, train_config, enn_config, direct_tensor_files, Predictor, device, seed_training, if_print = 1)
        wandb.log({"val_final_var_square_loss": meta_loss})





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command line arguments.")
    parser.add_argument("--config_file_path", type=str, help="Path to the JSON file containing the sweep configuration", default='config_sweep_adaptive_sampling.json')
    parser.add_argument("--project_name", type=str, help="WandB project name", default='adaptive_sampling_enn_regression')
    args = parser.parse_args()

    # Load sweep configuration from the JSON file
    with open(args.config_file_path, 'r') as config_file:
       config_params = json.load(config_file)



    # Initialize the sweep
    global ENTITY
    ENTITY = 'dm3766'
    global PROJECT_NAME
    PROJECT_NAME = args.project_name


    sweep_id = wandb.sweep(config_params, project=args.project_name, entity=ENTITY)
    # Run the agent
    wandb.agent(sweep_id, function=main_run_func)
    
   




