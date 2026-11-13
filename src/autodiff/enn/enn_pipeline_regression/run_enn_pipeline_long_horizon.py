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

import enn_pipeline_regression
import polyadic_sampler
from constant_network import ConstantValueNetwork
import wandb


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
    

        
        
        
        
        dataset_cfg = enn_pipeline_regression.DatasetConfig(direct_tensors_bool, csv_file_train, csv_file_test, csv_file_pool, y_column, shuffle)
        model_cfg = enn_pipeline_regression.ModelConfig(access_to_true_pool_y = access_to_true_pool_y, batch_size_query = batch_size_query, temp_k_subset = temp_k_subset, meta_opt_lr = meta_opt_lr, meta_opt_weight_decay = meta_opt_weight_decay, n_classes = n_classes)
        train_cfg = enn_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples, n_iter_noise = n_iter_noise, batch_size = batch_size) #temp_var_recall needs to be added as a new variable here i var recall setting
        enn_cfg = enn_pipeline_regression.ENNConfig(basenet_hidden_sizes = basenet_hidden_sizes, exposed_layers = exposed_layers, z_dim = z_dim, learnable_epinet_hiddens = learnable_epinet_hiddens, hidden_sizes_prior = hidden_sizes_prior, seed_base = seed_base, seed_learnable_epinet = seed_learnable_epinet, seed_prior_epinet = seed_prior_epinet, alpha = alpha, n_ENN_iter = n_ENN_iter, ENN_opt_lr = ENN_opt_lr, ENN_opt_weight_decay = ENN_opt_weight_decay, z_samples = z_samples, stdev_noise=stdev_noise)

        
        
        model_predictor = ConstantValueNetwork(constant_value=0.0, output_size=1).to(device)
        model_predictor.eval()



        for _ in range(5): #run 5 iterations
            if _ == 0: #1st iteration obtain data
                NN_weights, var_square_loss, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y = enn_pipeline_regression.experiment(dataset_cfg, model_cfg, train_cfg, enn_cfg, direct_tensor_files, model_predictor, device, seed_training, if_print = 1, if_return_nn_weights = 1)
            else:
                NN_weights, var_square_loss, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y  = enn_pipeline_regression.experiment_long_horizon(dataset_cfg, model_cfg, train_cfg, enn_cfg, direct_tensor_files, model_predictor, device, seed_training, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, if_print = 1, if_return_nn_weights = 1)

            _, indices = torch.topk(NN_weights, model_cfg.batch_size_query) #select top k indices
            #remaining index after top k values
            remaining_indices = list(set(list(range(pool_x.shape[0]))) - set(indices)) #needes to be checked

            #add those to training
            init_train_x = torch.cat((init_train_x, pool_x[indices, ]), 0)
            init_train_y = torch.cat((init_train_y, pool_y[indices ]), 0)

            #remove those points from pool 
            pool_x = pool_x[remaining_indices, ]
            pool_y = pool_y[remaining_indices]


        wandb.log({"val_final_var_square_loss": var_square_loss})





# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="This script processes command line arguments.")
#     parser.add_argument("--config_file_path", type=str, help="Path to the JSON file containing the sweep configuration", default='config_sweep_adaptive_sampling.json')
#     parser.add_argument("--project_name", type=str, help="WandB project name", default='adaptive_sampling_enn_regression')
#     args = parser.parse_args()

#     # Load sweep configuration from the JSON file
#     with open(args.config_file_path, 'r') as config_file:
#        config_params = json.load(config_file)



#     # Initialize the sweep
#     global ENTITY
#     ENTITY = 'dm3766'
#     global PROJECT_NAME
#     PROJECT_NAME = args.project_name


#     sweep_id = wandb.sweep(config_params, project=args.project_name, entity=ENTITY)
#     # Run the agent
#     wandb.agent(sweep_id, function=main_run_func)
    
   




