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

import gp_pipeline_regression
import polyadic_sampler_new as polyadic_sampler
from predictor_network import train_predictor, load_predictor
import wandb
import copy


n_samples_track = 1000

from variance_l_2_loss import var_l2_loss_estimator, l2_loss
from polyadic_sampler_new import CustomizableGPModel
from matplotlib import pyplot as plt

def plot_visualization(train_x, train_y, step, version = ''):
    if train_x.size(1) == 1: 
    
      fig2 = plt.figure()
      plt.scatter(train_x.to("cpu"),  train_y.to("cpu"), label='Train')

      wandb.log({"Acquired points at step"+str(step)+version: wandb.Image(fig2)})
      plt.close(fig2)

def posterior_visualization(model,x,step):
    if x.size(1) == 1:

        fig2 = plt.figure()
        posterior = model.likelihood(model(x))
        posterior_mean = posterior.mean
        posterior_std = torch.sqrt(posterior.variance)
        plt.scatter(x.to("cpu"),posterior_mean.detach().to("cpu").numpy())
        plt.scatter(x.squeeze().to("cpu"),posterior_mean.detach().to("cpu").numpy()-2*posterior_std.detach().to("cpu").numpy(),alpha=0.2)
        plt.scatter(x.squeeze().to("cpu"),posterior_mean.detach().to("cpu").numpy()+2*posterior_std.detach().to("cpu").numpy(),alpha=0.2)
        plt.title("Posterior at step "+str(step))
        wandb.log({"Posterior at step"+str(step): wandb.Image(fig2)})
        plt.close(fig2)



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
        no_horizons = config.no_horizons 
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


        csv_directory = config.csv_directory



        access_to_true_pool_y = config.access_to_true_pool_y    #true or false
        hyperparameter_tune = config.hyperparameter_tune   #true or false
        batch_size_query = config.batch_size_query
        temp_k_subset = config.temp_k_subset
        meta_opt_lr = config.meta_opt_lr
        meta_opt_weight_decay = config.meta_opt_weight_decay


        n_train_iter = config.n_train_iter
        n_samples = config.n_samples     #n_samples in variance calculation
        G_samples = config.G_samples     #G_samples in gradient average caluclation

        mean_constant = config.mean_constant
        length_scale = config.length_scale
        output_scale = config.output_scale
        noise_var =  config.noise_var
        parameter_tune_lr = config.parameter_tune_lr
        parameter_tune_weight_decay = config.parameter_tune_weight_decay
        parameter_tune_nepochs = config.parameter_tune_epochs
        stabilizing_constant =  config.stabilizing_constant 


        dataset_mean_constant =  config.dataset_mean_constant 
        dataset_length_scale =  config.dataset_length_scale
        dataset_output_scale =  config.dataset_output_scale
        dataset_noise_std  =  config.dataset_noise_std
       
        seed_dataset = config.seed_dataset 
        seed_training = config.seed_training        
        
        
        #print('load:', load)
        #print('load_project_name:', load_project_name)
        #print('load_artifact_name:', load_artifact_name)
        #print('save:', save)
        #print('save_project_name:', save_project_name)
        #print('save_artifact_name:', save_artifact_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        torch.manual_seed(seed_dataset)
        np.random.seed(seed_dataset)
        if device=="cuda":
            torch.cuda.manual_seed(seed_dataset) # Sets the seed for the current GPU
            torch.cuda.manual_seed_all(seed_dataset) # Sets the seed for all GPUs
        
        if csv_directory is not None:
            train_x = pd.read_csv(csv_directory+"train_x.csv")
            numpy_array_train_x = train_x.to_numpy()
            train_x= torch.tensor(numpy_array_train_x, dtype=torch.float32)[:,1:]

            train_y = pd.read_csv(csv_directory+"train_y.csv")
            numpy_array_train_y = train_y.to_numpy()
            train_y = (torch.tensor(numpy_array_train_y, dtype=torch.float32)[:,1]).squeeze()

            pool_x = pd.read_csv(csv_directory+"pool_x.csv")
            numpy_array_pool_x = pool_x.to_numpy()
            pool_x = torch.tensor(numpy_array_pool_x, dtype=torch.float32)[:,1:]

            pool_y = pd.read_csv(csv_directory+"pool_y.csv")
            numpy_array_pool_y = pool_y.to_numpy()
            pool_y = (torch.tensor(numpy_array_pool_y, dtype=torch.float32)[:,1]).squeeze()

            test_x = pd.read_csv(csv_directory+"test_x.csv")
            numpy_array_test_x = test_x.to_numpy()
            test_x = torch.tensor(numpy_array_test_x, dtype=torch.float32)[:,1:]

            test_y = pd.read_csv(csv_directory+"test_y.csv")
            numpy_array_test_y = test_y.to_numpy()
            test_y = (torch.tensor(numpy_array_test_y, dtype=torch.float32)[:,1]).squeeze()

            pool_sample_idx = torch.tensor(list(range(pool_x.shape[0])))
            test_sample_idx = torch.tensor(list(range(test_x.shape[0])))

            train_x = train_x.to(device)
            train_y = train_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            pool_x= pool_x.to(device)
            pool_y = pool_y.to(device)
            test_sample_idx = test_sample_idx.to(device)
            pool_sample_idx = pool_sample_idx.to(device)

            direct_tensor_files = (train_x, train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx)
        
        elif direct_tensors_bool:
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
            

        
        
        
        
        
        
        
        dataset_cfg = gp_pipeline_regression.DatasetConfig(direct_tensors_bool, csv_file_train, csv_file_test, csv_file_pool, y_column)
        model_cfg = gp_pipeline_regression.ModelConfig(access_to_true_pool_y = access_to_true_pool_y, hyperparameter_tune = hyperparameter_tune, batch_size_query = batch_size_query, temp_k_subset = temp_k_subset, meta_opt_lr = meta_opt_lr, meta_opt_weight_decay = meta_opt_weight_decay)
        train_cfg = gp_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples) #temp_var_recall is the new variable added here
        gp_cfg = gp_pipeline_regression.GPConfig(mean_constant=mean_constant, length_scale=length_scale, output_scale= output_scale, noise_var = noise_var, parameter_tune_lr = parameter_tune_lr, parameter_tune_weight_decay = parameter_tune_weight_decay, parameter_tune_nepochs = parameter_tune_nepochs, stabilizing_constant = stabilizing_constant)

        # Load or train the predictor network
        if PREDICTOR_PATH is not None:
            print(f"Loading pre-trained predictor from {PREDICTOR_PATH}")
            model_predictor = load_predictor(PREDICTOR_PATH, device=device)
        else:
            print("Training new predictor network...")
            model_predictor = train_predictor(
                train_x=train_x,
                train_y=train_y,
                device=device,
                input_dim=input_dim,
                hidden_dims=[64, 32],
                output_size=1,
                dropout=0.1,
                lr=1e-3,
                weight_decay=1e-4,
                n_epochs=100,
                batch_size=min(32, train_x.size(0)),
                val_split=0.2,
                early_stopping_patience=10,
                verbose=True
            )
        model_predictor.eval()



        torch.manual_seed(seed_training)
        np.random.seed(seed_training)
        if device=="cuda":
            torch.cuda.manual_seed(seed_training) # Sets the seed for the current GPU
            torch.cuda.manual_seed_all(seed_training) # Sets the seed for all GPUs


        mean_module_track = gpytorch.means.ConstantMean()
        base_kernel_track  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        likelihood_track  = gpytorch.likelihoods.GaussianLikelihood()

        mean_constant_track = gp_cfg.mean_constant
        length_scale_track  = gp_cfg.length_scale
        noise_var_track  = gp_cfg.noise_var
        output_scale_track  = gp_cfg.output_scale

        mean_module_track.constant = mean_constant_track
        base_kernel_track.base_kernel.lengthscale = length_scale_track 
        base_kernel_track.outputscale = output_scale_track 
        likelihood_track.noise_covar.noise = noise_var_track 


        gp_model_track  = CustomizableGPModel(train_x, train_y, mean_module_track , base_kernel_track , likelihood_track).to("cpu")
        gp_model_track.eval()
        likelihood_track.eval()
        # Create a CPU copy of the trained predictor for tracking
        model_predictor_cpu = copy.deepcopy(model_predictor).to("cpu")
        model_predictor_cpu.eval()

        mean_track, loss_track = var_l2_loss_estimator(gp_model_track, test_x.to("cpu"), model_predictor_cpu, "cpu", n_samples_track)

        mean_actual = l2_loss(test_x, test_y, model_predictor, (test_x).device)

        wandb.log({"var_square_loss_track": loss_track.item(), "l2_loss_track": mean_track.item(), "l2_loss_actual_track": mean_actual.item()})
    
        plot_visualization(train_x, train_y, -1)
        posterior_visualization(gp_model_track,test_x.to("cpu"),-1)
        for a in range(no_horizons):
            var_square_loss, NN_weights = gp_pipeline_regression.experiment(dataset_cfg, model_cfg, train_cfg, gp_cfg, direct_tensor_files, model_predictor, device, if_print = 0)
            wandb.log({"val_final_var_square_loss": var_square_loss})
            _, indices = torch.topk(NN_weights, model_cfg.batch_size_query) #select top k indices
            #remaining index after top k values
            remaining_indices = list(set(list(range(pool_x.shape[0]))) - set(indices)) #needes to be checked

            #add those to training
            train_x = torch.cat((train_x, pool_x[indices, ]), 0)
            train_y = torch.cat((train_y, pool_y[indices ]), 0)

            plot_visualization(pool_x[indices, ], pool_y[indices ], a,',pool')

            gp_model_track  = CustomizableGPModel(train_x, train_y, mean_module_track , base_kernel_track , likelihood_track ).to("cpu")
            gp_model_track.eval()
            likelihood_track.eval()

            mean_track, loss_track = var_l2_loss_estimator(gp_model_track, test_x.to("cpu"), model_predictor_cpu, "cpu", n_samples_track)
            mean_actual = l2_loss(test_x, test_y, model_predictor, (test_x).device)
            wandb.log({"var_square_loss_track": loss_track.item(), "l2_loss_track": mean_track.item(), "l2_loss_actual_track": mean_actual.item()})



             
            #remove those points from pool 
            pool_x = pool_x[remaining_indices, ]
            pool_y = pool_y[remaining_indices]
            pool_sample_idx = pool_sample_idx[remaining_indices]
            direct_tensor_files = (train_x, train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx)

            plot_visualization(train_x, train_y,a)
            posterior_visualization(gp_model_track,test_x.to("cpu"),a)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command line arguments.")
    parser.add_argument("--config_file_path", type=str, help="Path to the JSON file containing the sweep configuration", default='config_sweep.json')
    parser.add_argument("--project_name", type=str, help="WandB project name", default='adaptive_sampling_gp')
    parser.add_argument("--predictor_path", type=str, help="Path to pre-trained predictor (if not provided, will train new one). Can also be 'zero', 'const:0.0', or 'const:1.0'", default=None)
    args = parser.parse_args()
    #wandb.login()

    # Load sweep configuration from the JSON file
    with open(args.config_file_path, 'r') as config_file:
       config_params = json.load(config_file)



    # Initialize the sweep
    global ENTITY
    ENTITY = 'dm3766'
    global PROJECT_NAME
    PROJECT_NAME = args.project_name
    global PREDICTOR_PATH
    PREDICTOR_PATH = args.predictor_path


    sweep_id = wandb.sweep(config_params, project=args.project_name, entity=ENTITY)
    # Run the agent
    wandb.agent(sweep_id, function=main_run_func)
    
   




