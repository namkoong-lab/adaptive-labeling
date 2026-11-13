#!/user/bw2762/.conda/envs/testbed_2/bin/python
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

import gp_pipeline_regression_pg
import polyadic_sampler
from constant_network import ConstantValueNetwork
import wandb


def main_run_func():    
        
        no_train_points = 10
        no_test_points = 30
        no_pool_points = 20
        dataset_model_name = "GP"   #"GP" or "blr"
        no_anchor_points = 6
        input_dim = 1
        stdev_scale = 0.5
        stdev_pool_scale = 0.5
        scaling_factor = 1.0
        scale_by_input_dim = True
        gp_model_dataset_generation = "specify_own"
        #model = config.model
        stdev_blr_w = 0.1
        stdev_blr_noise = 0.01
        logits =  None
        if_logits = False     #true or false
        if_logits_only_pool = False    #true or false
        plot_folder = None    #none or string
        
        
        direct_tensors_bool = True  #true or false
        csv_file_train = None
        csv_file_test = None
        csv_file_pool = None
        y_column = None



        access_to_true_pool_y = True    #true or false
        hyperparameter_tune = False  #true or false
        batch_size_query = 5
        temp_k_subset = 0.1
        meta_opt_lr = 0.1
        meta_opt_weight_decay = 0


        n_train_iter = 1000
        n_samples = 100     #n_samples in variance calculation
        G_samples = 1000     #G_samples in gradient average caluclation
        learning_rate_PG = 1e-2
        weight_decay = 0

        mean_constant = 0.0
        length_scale = 1.0
        noise_std = 1e-2
        output_scale = 0.69

        dataset_mean_constant =  0.0
        dataset_length_scale = 1.0
        dataset_output_scale =  0.69
        dataset_noise_std  =  0.1
       
        seed_dataset = 0
        seed_training = 0   

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

            direct_tensor_files = (train_x, train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx)
        else:
            direct_tensor_files = None
            

        
        
        
        
        
        
        
        dataset_cfg = gp_pipeline_regression_pg.DatasetConfig(direct_tensors_bool, csv_file_train, csv_file_test, csv_file_pool, y_column)
        model_cfg = gp_pipeline_regression_pg.ModelConfig(access_to_true_pool_y = access_to_true_pool_y, hyperparameter_tune = hyperparameter_tune, batch_size_query = batch_size_query, temp_k_subset = temp_k_subset, meta_opt_lr = meta_opt_lr, meta_opt_weight_decay = meta_opt_weight_decay)
        train_cfg = gp_pipeline_regression_pg.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples, learning_rate_PG = learning_rate_PG, weight_decay = weight_decay) #temp_var_recall is the new variable added here
        # gp_cfg = gp_pipeline_regression_modified.GPConfig(length_scale=length_scale, output_scale= output_scale, noise_var = noise_var, parameter_tune_lr = parameter_tune_lr, parameter_tune_weight_decay = parameter_tune_weight_decay, parameter_tune_nepochs = parameter_tune_nepochs, stabilizing_constant = stabilizing_constant)
        gp_cfg = gp_pipeline_regression_pg.GPConfig(length_scale=length_scale, output_scale= output_scale, noise_std = noise_std, mean_constant = mean_constant)

        model_predictor = ConstantValueNetwork(constant_value=0.0, output_size=1).to(device)
        model_predictor.eval()



        torch.manual_seed(seed_training)
        np.random.seed(seed_training)
        if device=="cuda":
            torch.cuda.manual_seed(seed_training) # Sets the seed for the current GPU
            torch.cuda.manual_seed_all(seed_training) # Sets the seed for all GPUs
        
        
        var_square_loss = gp_pipeline_regression_pg.experiment(dataset_cfg, model_cfg, train_cfg, gp_cfg, direct_tensor_files, model_predictor, device, if_print = 1)
        # wandb.log({"val_final_var_square_loss": var_square_loss})





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command line arguments.")
    parser.add_argument("--config_file_path", type=str, help="Path to the JSON file containing the sweep configuration", default='config_sweep_pg.json')
    parser.add_argument("--project_name", type=str, help="WandB project name", default='adaptive_sampling_gp_pg')
    args = parser.parse_args()

    # Load sweep configuration from the JSON file
    # with open(args.config_file_path, 'r') as config_file:
    #    config_params = json.load(config_file)
    wandb.init()
    main_run_func()
    
    # # Initialize the sweep
    global ENTITY
    ENTITY = 'ym2865'
    global PROJECT_NAME
    PROJECT_NAME = args.project_name


    sweep_id = wandb.sweep(config_params, project=args.project_name, entity=ENTITY)
    # Run the agent
    wandb.agent(sweep_id, function=main_run_func)
    
   



