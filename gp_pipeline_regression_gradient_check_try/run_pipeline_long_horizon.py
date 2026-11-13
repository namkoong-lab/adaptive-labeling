from typing import Callable, NamedTuple
import numpy as np
import pandas as pd
import torch
import dataclasses
import gpytorch
import argparse
import json
import math
import warnings
warnings.filterwarnings('ignore')

import gp_pipeline_regression
import gp_pipeline_regression_pg
import polyadic_sampler_new as polyadic_sampler
from constant_network import ConstantValueNetwork
import wandb
import torch.nn.functional as F
import statistics


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



        access_to_true_pool_y = config.access_to_true_pool_y    #true or false
        hyperparameter_tune = config.hyperparameter_tune   #true or false
        batch_size_query = config.batch_size_query
        temp_k_subset = config.temp_k_subset
        meta_opt_lr = config.meta_opt_lr
        meta_opt_weight_decay = config.meta_opt_weight_decay


        n_train_iter = config.n_train_iter
        n_samples = config.n_samples     #n_samples in variance calculation
        G_samples = config.G_samples     #G_samples in gradient average caluclation


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
        seed_training_all = config.seed_training        
        
        
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
            direct_tensor_files_cpu = (train_x.to("cpu"), train_y.to("cpu"), pool_x.to("cpu"), pool_y.to("cpu"), test_x.to("cpu"), test_y.to("cpu"), pool_sample_idx.to("cpu"), test_sample_idx.to("cpu"))
        else:
            direct_tensor_files = None

        

        pool_size = pool_x.size(0)
        dataset_cfg = gp_pipeline_regression.DatasetConfig(direct_tensors_bool, csv_file_train, csv_file_test, csv_file_pool, y_column)
        model_cfg = gp_pipeline_regression.ModelConfig(access_to_true_pool_y = access_to_true_pool_y, hyperparameter_tune = hyperparameter_tune, batch_size_query = batch_size_query, temp_k_subset = temp_k_subset, meta_opt_lr = meta_opt_lr, meta_opt_weight_decay = meta_opt_weight_decay)
        train_cfg = gp_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples) #temp_var_recall is the new variable added here
        train_cfg_3 = gp_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=10) 
        gp_cfg = gp_pipeline_regression.GPConfig(length_scale=length_scale, output_scale= output_scale, noise_var = noise_var, parameter_tune_lr = parameter_tune_lr, parameter_tune_weight_decay = parameter_tune_weight_decay, parameter_tune_nepochs = parameter_tune_nepochs, stabilizing_constant = stabilizing_constant)

        model_predictor = ConstantValueNetwork(constant_value=0.0, output_size=1).to(device)
        model_predictor_cpu = ConstantValueNetwork(constant_value=0.0, output_size=1).to("cpu")
        model_predictor.eval()
        model_predictor_cpu.eval()



        #mean_module_track = gpytorch.means.ConstantMean()
        #base_kernel_track  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #likelihood_track  = gpytorch.likelihoods.GaussianLikelihood()


        #length_scale_track  = gp_cfg.length_scale
        #noise_var_track  = gp_cfg.noise_var
        #output_scale_track  = gp_cfg.output_scale

        #mean_module_track.constant = 0.0
        #base_kernel_track.base_kernel.lengthscale = length_scale_track 
        #base_kernel_track.outputscale = output_scale_track 
        #likelihood_track.noise_covar.noise = noise_var_track 


        #gp_model_track  = CustomizableGPModel(train_x, train_y, mean_module_track , base_kernel_track , likelihood_track).to("cpu")
        #gp_model_track.eval()
        #likelihood_track.eval()
        #model_predictor_dumi =  ConstantValueNetwork(constant_value=0.0, output_size=1).to("cpu")
        #model_predictor_dumi.eval()

        #mean_track, loss_track = var_l2_loss_estimator(gp_model_track, test_x.to("cpu"), model_predictor_dumi, "cpu", n_samples_track)

        #mean_actual = l2_loss(test_x, test_y, model_predictor, (test_x).device)

        #wandb.log({"var_square_loss_track": loss_track.item(), "l2_loss_track": mean_track.item(), "l2_loss_actual_track": mean_actual.item()})
            
        #plot_visualization(train_x, train_y, -1)
        #posterior_visualization(gp_model_track,test_x.to("cpu"),-1)
        
        seed_for_random_NN_weights = 101
        torch.manual_seed(seed_for_random_NN_weights)
        np.random.seed(seed_for_random_NN_weights)
        print("device",device)

        if device=="cuda":
             torch.cuda.manual_seed(seed_for_random_NN_weights) # Sets the seed for the current GPU
             torch.cuda.manual_seed_all(seed_for_random_NN_weights) # Sets the seed for all GPUs

        NN_weights_to_be_cloned =  torch.rand([pool_size], device=device, requires_grad=True)


        torch.manual_seed(seed_training_all)
        np.random.seed(seed_training_all)
        print("device",device)

        if device=="cuda":
             torch.cuda.manual_seed(seed_training_all) # Sets the seed for the current GPU
             torch.cuda.manual_seed_all(seed_training_all) # Sets the seed for all GPUs

        random_tensor = 0
        if random_tensor == 1:
            NN_weights_1 = NN_weights_to_be_cloned.clone().detach().to("cpu").requires_grad_(True)
            NN_weights_2 = NN_weights_to_be_cloned.clone().detach().to("cpu").requires_grad_(True)
            NN_weights_3 = NN_weights_to_be_cloned.clone().detach().to("cpu").requires_grad_(True)
        else:
            NN_weights_1 = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device=device)
            NN_weights_2 = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device="cpu")
            NN_weights_3 = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device="cpu")


        #print("NN_weights",NN_weights)
            
        var_square_loss_3, NN_weights_3, grad_store_3 = gp_pipeline_regression_pg.experiment(dataset_cfg, model_cfg, train_cfg_3, gp_cfg, direct_tensor_files_cpu, model_predictor_cpu, "cpu", NN_weights_3, if_print = 1)
        print("grad_store_3",grad_store_3)
        original_grad_store_3 = grad_store_3.clone()
        true_2000_tensor_squared_mean_all = []
        squared_true_tensor_2000 = original_grad_store_3**2
        mean_squared_true_tensor_2000 = torch.mean(squared_true_tensor_2000)
        true_2000_tensor_squared_mean_all.append(mean_squared_true_tensor_2000.item())
        wandb.log({"true_2000_tensor_squared_mean": mean_squared_true_tensor_2000.item()})
        print("original_grad_store_3", original_grad_store_3)
        print("original_grad_store_3_sum", original_grad_store_3.sum())
        top5_values, top5_indices = torch.topk(original_grad_store_3, 5)
        least5_values, least5_indices = torch.topk(original_grad_store_3, 5, largest=False)
        print("original_grad_store_3_top_5", top5_values)
        print("original_grad_store_3_top_5_sum", top5_values.sum())
        print("top_5_indices_original_grad_store_3", top5_indices)
        print("original_grad_store_3_least_5", least5_values)
        print("original_grad_store_3_least_5_sum", least5_values.sum())
        print("least_5_indices_original_grad_store_3", least5_indices)


        for i in range(0):
            torch.manual_seed(i)
            np.random.seed(i)
            if device=="cuda":
                torch.cuda.manual_seed(i) # Sets the seed for the current GPU
                torch.cuda.manual_seed_all(i) # Sets the seed for all GPUs
            NN_weights_3 = NN_weights_1.clone().detach().to("cpu").requires_grad_(True)
            #print("NN_weights_3_grad",NN_weights_3.grad)
            print("iteration",i)
            var_square_loss_3, NN_weights_3, grad_store_4 = gp_pipeline_regression_pg.experiment(dataset_cfg, model_cfg, train_cfg_3, gp_cfg, direct_tensor_files_cpu, model_predictor_cpu, "cpu", NN_weights_3, if_print = 1)
            
            cos_sim_true = F.cosine_similarity(grad_store_3, grad_store_4, dim=0)
            cos_sim_true_original = F.cosine_similarity(original_grad_store_3, grad_store_4, dim=0)
            mse_loss = torch.nn.MSELoss()
            mse_true_original = mse_loss(grad_store_4, original_grad_store_3)

            grad_store_3 += grad_store_4
            #print("grad_store_4",grad_store_4)
            squared_true_tensor_2000 = grad_store_4**2
            mean_squared_true_tensor_2000 = torch.mean(squared_true_tensor_2000)
            true_2000_tensor_squared_mean_all.append(mean_squared_true_tensor_2000.item())
            

            wandb.log({"cos_sim_true": cos_sim_true.item(), "cos_sim_true_original": cos_sim_true_original.item(), "mse_true_original": mse_true_original.item(), "true_2000_tensor_squared_mean": mean_squared_true_tensor_2000.item()})


        #cos_sim_pathwise_average = statistics.mean(cos_sim_pathwise_all)
        #grad_store_3 = grad_store_3/2.0
        #mean_squared_true_average = statistics.mean(true_2000_tensor_squared_mean_all)
        #wandb.log({"mean_squared_true_average_all": mean_squared_true_average})
        #print("grad_store_3",grad_store_3)
        #mse_true_0 = mse_loss(grad_store_3, original_grad_store_3)
        #mse_true_25 = mse_loss(grad_store_3, grad_store_4)
        #mse_true_dumi = mse_loss(grad_store_3, grad_store_3) 
        
        #wandb.log({"mse_true_0": mse_true_0.item(), "mse_true_25": mse_true_25.item(), "mse_true_dumi": mse_true_dumi.item()})
        #print("grad_store_3",grad_store_3)
        #print("grad_store_3_sum",grad_store_3.sum())
        #top5_values, top5_indices = torch.topk(grad_store_3, 5)
        #least5_values, least5_indices = torch.topk(grad_store_3, 5, largest=False)
        #print("grad_store_3_top_5", top5_values)
        #print("grad_store_3_top_5_sum", top5_values.sum())
        #print("top_5_indices_grad_store_3", top5_indices)
        #print("grad_store_3_least_5", least5_values)
        #print("grad_store_3_least_5_sum", least5_values.sum())
        #print("least_5_indices_grad_store_3", least5_indices)
        #squared_true_tensor_50000 = grad_store_3**2
        #mean_squared_true_tensor_50000 = torch.mean(squared_true_tensor_50000)
        #print("mean_squared_true_tensor",mean_squared_true_tensor_50000)
        #wandb.log({"mean_squared_true_tensor":mean_squared_true_tensor_50000.item()})


        #THIS WAS FOR THE UNIFORM theta
        grad_store_3 = torch.tensor([ 1.0091e-05,  5.1621e-06,  5.4484e-06,  1.3711e-05,  6.8529e-06, 5.5476e-06, -3.3553e-06,  2.0412e-06,  7.7777e-06,  5.9872e-06,
         8.6929e-06,  5.1000e-06,  1.1590e-05,  3.5296e-06,  3.5442e-06,
         6.5053e-06,  1.3533e-05,  2.2765e-06,  7.3215e-06, -1.2364e-06,
         1.1508e-05,  1.2640e-05,  1.0101e-05,  6.9115e-06,  1.2244e-06,
         1.0446e-05,  3.9194e-06,  5.4225e-06,  1.1442e-05,  1.0968e-05,
         1.2006e-05,  9.2130e-06,  5.5019e-06,  6.7208e-06,  1.2341e-05,
         2.8967e-06,  5.7915e-06,  6.4197e-06,  8.4778e-06,  7.1948e-06,
         3.3346e-06,  8.7191e-06,  1.0328e-05,  1.0374e-05,  2.4181e-06,
         1.0428e-05,  9.8739e-06,  7.6726e-06,  5.8004e-06,  5.2061e-06,
         5.1730e-06,  5.9352e-06,  1.2592e-05,  6.2522e-06,  5.7481e-06,
         1.1321e-05,  5.8230e-06,  1.0426e-05,  9.7512e-06,  7.2481e-06,
         9.6744e-06,  5.8906e-06,  9.4117e-06,  1.1705e-05,  8.5040e-06,
         7.4388e-06,  5.4441e-06,  9.4416e-06,  7.7156e-06,  6.0473e-06,
         8.6413e-06,  6.1082e-06,  3.6541e-06,  1.4875e-06,  4.3230e-06,
         7.0114e-06,  8.2788e-06,  9.3740e-06,  1.2029e-05,  6.9282e-06,
         7.4646e-06,  6.2608e-06,  5.2349e-06,  9.7946e-06,  6.5676e-06,
         6.8594e-06,  9.0127e-07,  3.2646e-06,  1.1330e-05,  2.1315e-06,
         3.9046e-07,  1.3956e-05,  4.4156e-06,  8.1396e-06,  9.4108e-06,
         5.8022e-06,  1.3860e-05,  3.7008e-06,  8.6308e-06,  9.3018e-06,
         7.6620e-06,  1.2993e-05,  4.5671e-06,  9.9625e-06,  1.3585e-05,
         1.1210e-05,  3.3526e-06,  1.0428e-05, -2.5072e-06,  8.8680e-06,
        -4.3321e-07,  2.7219e-06,  3.3821e-06,  8.8003e-06,  8.8279e-06,
         9.4079e-06,  1.0034e-05,  3.8772e-06,  9.1365e-06, -2.2467e-06,
         7.8882e-06,  1.1691e-05,  1.0003e-05,  8.8371e-06,  9.0246e-06,
         6.3941e-06,  3.1060e-06, -1.7330e-07,  8.4273e-06,  9.8546e-06,
         1.2555e-05,  9.7536e-06,  7.5131e-06,  6.2778e-06,  7.6661e-06,
         1.1612e-05,  1.2856e-05,  8.3001e-06,  4.0028e-06,  7.5644e-06,
         7.7990e-06,  1.3041e-05,  2.3462e-06,  6.4466e-06,  1.2035e-05,
         7.5688e-06,  7.4821e-06,  1.1844e-05,  4.0536e-06, -3.2416e-07,
         1.1085e-05,  9.5970e-06,  9.8068e-06,  6.3437e-06,  1.4762e-05,
         1.2701e-05,  2.5688e-06,  7.4291e-06,  4.2813e-06,  1.1502e-06,
         9.9551e-06,  4.0108e-06,  5.0211e-06,  9.9045e-06,  1.8359e-06,
         1.7234e-05,  9.6546e-06,  1.1085e-05,  6.7116e-06,  5.9046e-06,
         4.6413e-07,  7.1366e-06,  7.3422e-06, -1.7952e-06,  5.7753e-06,
         4.9351e-06,  5.5553e-06,  3.0948e-06,  1.4629e-05,  7.7853e-06,
         6.1684e-06,  6.4813e-06,  4.8267e-06,  7.1991e-06,  1.1016e-05,
         4.5244e-06,  5.6953e-06,  2.9176e-06,  1.0444e-05,  9.9211e-06,
         3.2359e-06,  1.4354e-05,  5.8292e-06,  6.8665e-06,  6.9200e-06,
         1.2518e-05,  5.4997e-06,  4.2977e-06,  4.6615e-06,  3.5319e-06,
         5.4723e-06,  9.3404e-06,  6.6937e-06,  2.6776e-06,  9.8855e-06,
         9.0317e-06,  6.0581e-06,  7.8877e-06,  1.6786e-06,  9.6758e-06,
         2.0855e-06,  3.7125e-06,  8.5093e-06,  8.0695e-06,  2.5829e-06,
         9.5037e-06,  3.1335e-06,  1.2413e-05,  1.8006e-05,  7.2366e-06,
         5.4874e-06,  3.0685e-06,  4.5629e-06,  3.1613e-06,  9.4716e-06,
         9.0320e-07,  6.7879e-06,  1.1141e-05,  7.3577e-06,  1.2231e-05,
         6.4285e-06,  3.4280e-06,  9.5247e-06,  1.3910e-05,  3.2782e-06,
         5.5857e-06,  7.2796e-06,  1.1478e-05,  1.1482e-05,  1.2742e-05,
         4.6632e-06,  8.4440e-06,  1.1728e-05, -8.4589e-07,  9.8784e-06,
         4.5023e-06,  5.4657e-06,  1.2412e-06,  7.8435e-06,  5.6159e-06,
         1.3155e-06, -6.9860e-07,  1.0393e-05,  8.0883e-06,  4.7936e-06,
         2.6424e-06,  8.4897e-06,  4.5335e-06,  2.2180e-06,  1.9033e-06,
        -1.5164e-05, -1.0150e-05, -1.2179e-05, -1.4312e-05, -1.8604e-05,
        -2.0335e-05, -1.6406e-05, -1.9093e-05, -1.0766e-05, -1.4900e-05,
        -6.2428e-06,  2.2425e-06, -5.1662e-06, -1.0777e-06, -2.3200e-06,
        -1.6183e-05, -9.8660e-06, -1.8452e-05, -1.0680e-05, -2.7863e-05,
        -2.8104e-05, -2.2092e-05, -2.7777e-05, -3.0885e-05, -3.0009e-05,
        -3.6393e-05, -3.6500e-05, -3.0752e-05, -4.0890e-05, -3.1310e-05,
        -3.6278e-05, -4.1481e-05, -3.5694e-05, -3.3560e-05, -3.6122e-05,
        -1.8885e-05, -1.5020e-05, -2.1950e-05, -2.6404e-05, -1.6145e-05,
        -1.2507e-06,  6.1286e-08,  3.9645e-06, -3.4477e-06, -2.2972e-06,
        -1.6542e-05, -1.2180e-05, -1.5759e-05, -1.9949e-05, -2.1355e-05,
        -3.2717e-05, -2.4914e-05, -4.0310e-05, -2.2791e-05, -4.3339e-05,
        -3.2040e-05, -3.6960e-05, -3.7622e-05, -3.8042e-05, -3.2987e-05,
        -3.3575e-05, -3.6460e-05, -3.2254e-05, -3.1221e-05, -3.3055e-05,
        -3.7596e-05, -3.8783e-05, -3.9178e-05, -3.6840e-05, -4.2160e-05,
        -3.7264e-05, -4.0784e-05, -3.7251e-05, -3.9908e-05, -3.5148e-05,
        -2.2967e-05, -2.9254e-05, -2.9943e-05, -3.3173e-05, -2.6222e-05,
        -1.7040e-05, -1.3541e-05, -2.3013e-05, -1.6295e-05, -2.0903e-05,
        -4.8139e-06, -1.1255e-05, -6.2025e-06, -1.5696e-05, -8.3706e-06,
        -2.0308e-06, -1.4617e-05, -3.6639e-06, -4.8064e-06, -1.7601e-05,
        -2.1309e-05, -2.1630e-05, -1.4410e-05, -1.6774e-05, -1.8606e-05,
        -3.6107e-05, -3.7739e-05, -3.5465e-05, -3.4299e-05, -4.4838e-05,
        -3.6016e-05, -3.6701e-05, -3.4167e-05, -3.1986e-05, -3.3834e-05,
        -9.4487e-06, -1.8555e-05, -1.6352e-05, -9.7780e-06, -1.6795e-05,
         1.0427e-05,  1.1736e-05,  6.7348e-06,  9.2219e-06,  6.1366e-06,
         3.7887e-06,  6.0405e-06,  7.6796e-06,  4.4691e-06,  6.9528e-06,
         6.0372e-06,  9.4328e-06,  7.0846e-06,  6.2949e-06,  1.9679e-06,
         9.8778e-06,  1.2482e-06,  9.8808e-06,  2.0759e-06,  1.1785e-05,
         7.3732e-06,  4.6676e-06,  6.2691e-06,  1.1358e-05,  4.4641e-06,
        -7.3485e-07,  3.5062e-06,  1.0068e-05,  6.6266e-06,  6.4018e-06,
         5.7272e-06,  1.2912e-05,  7.0262e-06,  5.8361e-06,  6.3219e-06,
         3.3211e-06,  1.0799e-05,  1.0332e-05,  6.3045e-06,  7.9877e-06,
         6.1080e-06,  7.1590e-06,  4.8063e-06,  4.9625e-06,  2.2610e-07,
         6.9108e-06,  8.3023e-06,  2.1138e-07,  3.0987e-06,  4.7838e-06,
         1.4115e-05,  4.3681e-06,  8.7151e-06,  4.2340e-06, -8.9750e-07,
         1.0200e-05,  5.7348e-06,  1.3196e-05,  4.2477e-06,  5.2068e-06,
         9.3818e-06,  1.0976e-05,  4.2039e-06, -3.3673e-07,  5.4143e-06,
         1.2986e-05,  1.0936e-05,  1.2727e-05,  1.0189e-05,  7.7396e-06,
         9.4526e-06,  7.4205e-06,  6.1896e-06,  4.5805e-06,  5.9653e-06,
         2.9835e-06,  7.1500e-06,  7.4299e-06,  9.9583e-06,  5.4505e-06,
         2.9455e-06,  8.6557e-06,  5.6782e-06,  5.0042e-06,  1.4555e-06,
         3.2407e-06,  5.3408e-06,  7.2567e-07,  8.2815e-06,  1.3775e-05,
         8.3194e-06,  3.9594e-06,  9.7867e-06,  1.2820e-05,  1.2585e-05,
         2.5928e-06,  3.9390e-06,  1.2047e-05,  8.1388e-06,  4.1226e-06,
         4.0715e-06,  5.0191e-06,  4.4546e-06,  1.0412e-05,  9.0413e-06,
        -5.5466e-07,  7.2309e-06,  1.0365e-05,  9.8645e-06,  7.4108e-06,
         3.4938e-06,  6.5103e-06,  6.9390e-06,  1.2421e-05,  8.0993e-06,
         8.2332e-06,  4.6508e-06,  5.0209e-06,  2.1762e-06,  4.8795e-06,
         7.8691e-06,  9.4106e-06,  6.4270e-06,  3.7157e-06,  1.0756e-07])

        # grad_store_3 = torch.tensor([-4.9268e-06,  5.0781e-06, -6.2682e-06,  1.6725e-05,  1.3748e-05,
        # -3.4185e-06,  2.5553e-06,  7.3544e-07, -7.6364e-06,  1.1181e-05,
        #  5.5767e-06,  8.3142e-06,  4.0451e-05,  7.2437e-06,  5.0810e-06,
        #  2.3270e-05,  1.6870e-05,  8.7374e-06,  2.2138e-05,  2.4078e-05,
        #  2.1028e-05, -4.7899e-06,  5.7471e-06, -5.1739e-06, -1.3165e-08,
        # -5.3864e-07,  7.1939e-06,  1.4616e-05,  6.7811e-06,  2.4577e-05,
        #  2.8577e-06, -4.3877e-06,  1.4310e-05, -3.0963e-06,  1.7872e-05,
        #  2.1318e-05, -8.4721e-06,  3.8305e-06,  7.1876e-06,  1.8557e-06,
        # -9.1078e-06, -1.1330e-05,  2.9915e-05,  1.7075e-05,  8.0275e-07,
        #  2.6578e-05,  2.9870e-06,  1.1674e-05,  3.0211e-05,  1.0246e-06,
        #  2.5855e-06,  1.6593e-05,  5.6439e-06,  3.6185e-06,  1.4533e-05,
        #  1.5440e-05, -1.1866e-05, -3.0458e-06,  6.0348e-06,  2.0827e-05,
        # -8.7096e-06,  2.3686e-05, -1.6574e-05,  2.1103e-05, -3.0344e-06,
        # -9.1431e-06,  1.2888e-05, -6.1572e-06, -5.8990e-07, -5.5415e-06,
        #  5.8276e-07, -8.3944e-06,  4.4164e-06,  2.9989e-06, -3.3390e-06,
        # -2.3476e-06,  2.2631e-05,  3.4300e-05,  1.8049e-06,  1.3344e-05,
        # -2.1892e-07,  1.4380e-05, -6.2339e-06,  1.2820e-05, -1.7108e-06,
        #  4.2249e-06,  2.0818e-06, -4.8640e-06,  1.0446e-05,  2.7220e-05,
        #  1.0328e-05,  1.3449e-05, -9.9253e-06,  6.5264e-07, -3.8542e-06,
        #  9.1577e-06,  6.5740e-06,  6.9022e-06,  3.3627e-05, -6.8001e-06,
        #  3.1284e-05,  1.5587e-05,  1.5493e-05,  8.5306e-06,  1.7253e-05,
        #  1.6195e-05,  8.9391e-06,  1.4529e-05, -7.1145e-06, -5.2335e-06,
        # -1.9109e-06, -6.5011e-06,  2.4807e-05,  1.3265e-05, -4.9226e-06,
        #  2.0338e-05,  6.5919e-06,  1.5665e-05, -1.0414e-06,  4.1779e-06,
        #  3.8660e-06,  2.2437e-06,  2.3056e-05,  7.3351e-06, -1.4767e-06,
        #  7.7314e-06,  9.2751e-06,  6.2039e-06,  2.0661e-05,  6.5987e-06,
        # -3.0049e-06,  7.7881e-06,  1.6534e-06,  2.0566e-05,  1.2117e-05,
        #  2.8001e-05,  2.2019e-05,  1.1910e-05,  1.5701e-05,  1.2413e-05,
        #  4.2316e-07, -7.9072e-07, -1.6363e-05,  3.1540e-06,  4.3165e-06,
        # -4.0075e-07,  1.5288e-05,  1.0435e-06,  1.3196e-05,  1.9599e-06,
        #  2.4228e-06,  3.7160e-05,  1.5202e-05, -8.2216e-07, -1.6940e-06,
        #  1.6498e-05,  2.0644e-05, -9.9310e-07, -4.5384e-06, -5.3949e-06,
        #  3.0808e-06,  6.0446e-06,  3.6661e-06,  1.3466e-05,  3.1278e-06,
        #  1.5888e-05, -1.7799e-07,  1.0118e-06,  1.5464e-05,  1.3364e-05,
        #  3.9441e-06, -1.6853e-06, -3.4340e-06,  2.0678e-06,  4.2475e-06,
        # -4.5604e-06, -1.7171e-06,  1.3030e-05, -4.0326e-06, -8.1475e-06,
        #  3.0433e-05, -1.9261e-06,  2.0126e-05,  3.3500e-06,  9.3424e-06,
        # -3.4950e-06,  2.4970e-06,  1.4541e-05,  2.7969e-05, -1.2748e-05,
        #  1.1553e-05,  8.8994e-06, -9.6627e-06,  4.2429e-06,  7.1362e-06,
        #  1.6568e-08,  5.0007e-06, -9.7035e-06, -1.1994e-05,  7.0282e-06,
        #  5.9676e-06,  1.7160e-05,  5.6783e-06,  1.1907e-05,  4.4310e-06,
        #  2.2659e-05,  1.2363e-05,  9.4158e-06,  1.0695e-05,  8.9299e-06,
        # -9.1425e-06, -5.1721e-06,  1.2280e-05, -1.9468e-06, -3.8769e-07,
        #  1.1872e-05, -6.2006e-06,  2.4551e-05,  2.7426e-05,  3.5505e-07,
        # -2.5415e-06,  1.5387e-05, -3.7569e-06, -6.2939e-08, -1.3387e-05,
        # -5.8313e-07,  9.8123e-06, -1.0414e-05,  2.3187e-05,  1.7618e-05,
        #  9.6712e-07, -1.1935e-05, -8.1560e-07,  2.0737e-05,  3.2939e-05,
        #  1.3186e-05,  1.6312e-05,  2.4558e-05,  1.9814e-05,  1.1963e-05,
        #  8.1048e-06,  3.6652e-05,  3.6452e-06, -5.5549e-06,  1.3918e-05,
        # -2.1067e-06,  2.5936e-05, -2.5633e-06,  1.1288e-05, -9.5982e-09,
        #  1.3505e-05,  9.3936e-07,  5.8110e-06,  4.7160e-06,  3.3520e-06,
        # -1.0581e-05, -3.1603e-06,  3.6315e-06, -1.7512e-05, -9.1369e-06,
        # -3.9783e-05, -1.5446e-05, -1.3123e-05, -2.4343e-05, -1.9791e-05,
        # -2.6340e-06, -2.1784e-05, -1.8473e-05, -9.6167e-06, -4.9428e-06,
        #  7.7228e-07,  2.2381e-05, -4.1840e-06, -2.9046e-06,  5.8843e-06,
        # -3.9297e-05, -1.0981e-05, -1.3143e-05, -2.7349e-05, -4.3222e-05,
        # -2.1868e-05, -2.9566e-05, -1.1085e-05, -1.2995e-05, -4.5470e-05,
        # -5.1668e-05, -4.5903e-05, -2.5061e-05, -4.2227e-05, -4.8389e-05,
        # -4.5079e-05, -4.5929e-05, -1.9044e-05, -4.9683e-05, -2.4143e-05,
        # -1.2058e-05, -1.1060e-05, -4.4721e-05, -4.3579e-05, -8.9863e-06,
        #  7.9998e-06, -2.0387e-06,  7.4579e-06, -1.0527e-05,  8.1319e-06,
        # -1.7633e-05, -6.9885e-06, -2.4818e-05, -1.9707e-05, -2.2785e-05,
        # -3.1582e-05, -2.7821e-05, -3.0378e-05, -2.1660e-05, -2.8334e-05,
        # -5.5142e-05, -4.6025e-05, -3.2748e-05, -2.7535e-05, -3.1194e-05,
        # -4.7719e-05, -4.5112e-05, -6.9605e-05, -2.4480e-05, -3.7247e-05,
        # -4.6935e-05, -3.7275e-05, -5.3685e-05, -3.3047e-05, -5.9549e-05,
        # -3.1635e-05, -6.5277e-05, -2.5189e-05, -9.3342e-06, -1.2155e-05,
        # -8.8871e-06, -2.4244e-05, -2.9306e-05, -3.0303e-05, -3.3030e-05,
        # -6.9335e-06, -7.1716e-06, -2.7739e-05, -2.3818e-05, -1.2628e-05,
        # -4.5738e-06, -8.7257e-06,  6.2463e-06, -2.1139e-05, -1.1007e-05,
        #  3.1755e-06, -3.2625e-05,  3.6120e-06,  5.1089e-06, -2.0058e-05,
        # -2.0029e-05, -1.9734e-05,  1.0380e-05, -5.3309e-06, -1.4724e-05,
        # -3.8858e-05, -2.9801e-05, -4.7096e-05, -2.1246e-05, -2.9348e-05,
        # -4.5216e-05, -3.5846e-05, -3.6078e-05, -3.5710e-05, -1.6176e-05,
        # -1.6268e-05, -6.3533e-06, -2.5215e-05, -1.8474e-05, -9.7166e-06,
        #  3.0771e-05,  8.0962e-06,  9.2869e-06,  5.1444e-07,  2.2685e-06,
        #  4.6952e-06,  1.3443e-06,  1.1062e-05,  4.8193e-06,  1.9283e-05,
        # -3.9334e-06,  5.4952e-07, -8.1564e-07,  2.1259e-05,  1.1816e-05,
        # -1.1489e-05,  1.4472e-05,  5.1378e-06, -1.7538e-05,  1.8445e-05,
        #  1.3621e-05, -6.3095e-06,  1.3856e-05,  2.2711e-05,  1.2565e-05,
        # -2.9694e-06,  2.1709e-05,  1.7297e-05, -1.5431e-05,  3.9459e-06,
        #  9.9599e-06,  1.5131e-05,  1.5294e-05, -8.4638e-07,  2.1887e-05,
        #  1.0443e-06,  2.9532e-05,  1.2338e-05,  1.7238e-05, -1.4483e-05,
        # -4.0877e-06, -5.0672e-06, -1.1539e-05,  1.4977e-05, -5.3804e-06,
        #  1.1251e-05,  9.3606e-06, -3.5864e-06, -6.2446e-06,  8.6154e-06,
        #  2.2487e-05,  1.6177e-05,  1.2166e-05,  1.8482e-05,  1.7398e-05,
        #  7.2741e-06,  6.9992e-06,  3.4385e-06,  2.7376e-05, -7.7564e-06,
        #  3.4880e-06,  6.8559e-06, -3.7078e-06, -1.9340e-05,  5.9193e-06,
        #  1.5940e-05,  1.2535e-06,  2.1715e-05,  4.1642e-05,  7.5680e-06,
        # -1.1614e-05,  2.2042e-05,  2.5160e-05,  7.8051e-06, -9.9449e-07,
        #  1.2918e-05, -3.9167e-06,  1.7053e-05, -7.1441e-06,  1.5269e-05,
        # -3.1318e-06,  1.9837e-05, -5.1401e-06,  5.4052e-06,  5.3096e-06,
        #  3.7047e-06,  2.6513e-06,  6.5970e-06,  2.1909e-05,  1.1520e-05,
        #  2.1514e-05,  2.9950e-05,  1.2290e-05,  1.9631e-05,  7.4471e-06,
        #  3.0980e-06, -9.2133e-06,  2.1991e-06, -1.5902e-06,  6.3706e-06,
        #  1.1799e-05,  7.6936e-06,  9.1346e-06,  2.1954e-05, -7.0202e-06,
        #  1.0698e-06, -2.7939e-06,  6.7303e-06,  1.4559e-05,  3.4060e-05,
        #  1.6582e-05,  1.1503e-05,  1.3331e-05,  8.3835e-06,  7.3229e-07,
        #  9.9011e-06, -4.9325e-06,  9.8001e-07, -1.1236e-05,  5.9524e-07,
        #  4.6015e-06,  1.8524e-05, -1.3707e-05,  2.7586e-06, -6.4458e-06])


        print("grad_store_3",grad_store_3)
        print("grad_store_3_sum",grad_store_3.sum())
        top5_values, top5_indices = torch.topk(grad_store_3, 5)
        least5_values, least5_indices = torch.topk(grad_store_3, 5, largest=False)
        print("grad_store_3_top_5", top5_values)
        print("grad_store_3_top_5_sum", top5_values.sum())
        print("top_5_indices_grad_store_3", top5_indices)
        print("grad_store_3_least_5", least5_values)
        print("grad_store_3_least_5_sum", least5_values.sum())
        print("least_5_indices_grad_store_3", least5_indices)
        squared_true_tensor_50000 = grad_store_3**2
        mean_squared_true_tensor_50000 = torch.mean(squared_true_tensor_50000)
        print("mean_squared_true_tensor",mean_squared_true_tensor_50000)
        wandb.log({"mean_squared_true_tensor":mean_squared_true_tensor_50000.item()})




        
        
        
        
        
        
        
 

        #for G_samples in [8,1000]:
        for G_samples in [8,1000]:
            train_cfg = gp_pipeline_regression.TrainConfig(n_train_iter = n_train_iter, n_samples = n_samples, G_samples=G_samples) #temp_var_recall is the new variable added here
            cos_sim_pathwise_all = []
            cos_sim_reinforce_all = []
            mse_pathwise_all = []
            mse_reinforce_all = []
            pathwise_tensor_squared_mean_all = []
            reinforce_tensor_squared_mean_all = []

            for seed_training in range(200):
                
        
               
                #pool_size = pool_x.size(0)

                torch.manual_seed(seed_training*1000)
                np.random.seed(seed_training*1000)
                

                if device=="cuda":
                    torch.cuda.manual_seed(seed_training*1000) # Sets the seed for the current GPU
                    torch.cuda.manual_seed_all(seed_training*1000) # Sets the seed for all GPUs

                print("seed_training", seed_training*1000)

                if random_tensor == 1:
                    NN_weights_a = NN_weights_to_be_cloned.clone().detach().to(device).requires_grad_(True)
                    NN_weights_2 = NN_weights_to_be_cloned.clone().detach().to("cpu").requires_grad_(True)
                    NN_weights_3 = NN_weights_to_be_cloned.clone().detach().to("cpu").requires_grad_(True)
                else:
                    NN_weights_a = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device=device)
                    NN_weights_2 = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device="cpu")
                    NN_weights_3 = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device="cpu")

                



                #print("NN_weights",NN_weights)
                #print("NN_weights_2",NN_weights_2)
                #print("NN_weights_3",NN_weights_3)
                    

                


                #mean_module_track = gpytorch.means.ConstantMean()
                #base_kernel_track  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                #likelihood_track  = gpytorch.likelihoods.GaussianLikelihood()


                #length_scale_track  = gp_cfg.length_scale
                #noise_var_track  = gp_cfg.noise_var
                #output_scale_track  = gp_cfg.output_scale

                #mean_module_track.constant = 0.0
                #base_kernel_track.base_kernel.lengthscale = length_scale_track 
                #base_kernel_track.outputscale = output_scale_track 
                #likelihood_track.noise_covar.noise = noise_var_track 


                #gp_model_track  = CustomizableGPModel(train_x, train_y, mean_module_track , base_kernel_track , likelihood_track).to("cpu")
                #gp_model_track.eval()
                #likelihood_track.eval()
                #model_predictor_dumi =  ConstantValueNetwork(constant_value=0.0, output_size=1).to("cpu")
                #model_predictor_dumi.eval()

                #mean_track, loss_track = var_l2_loss_estimator(gp_model_track, test_x.to("cpu"), model_predictor_dumi, "cpu", n_samples_track)

                #mean_actual = l2_loss(test_x, test_y, model_predictor, (test_x).device)

                #wandb.log({"var_square_loss_track": loss_track.item(), "l2_loss_track": mean_track.item(), "l2_loss_actual_track": mean_actual.item()})
            
                #plot_visualization(train_x, train_y, -1)
                #posterior_visualization(gp_model_track,test_x.to("cpu"),-1)
                for a in range(no_horizons):
                    var_square_loss, NN_weights_a, grad_store_1 = gp_pipeline_regression.experiment(dataset_cfg, model_cfg, train_cfg, gp_cfg, direct_tensor_files, model_predictor, device, NN_weights_a, if_print = 0)
                    var_square_loss_2, NN_weights_2, grad_store_2 = gp_pipeline_regression_pg.experiment(dataset_cfg, model_cfg, train_cfg, gp_cfg, direct_tensor_files_cpu, model_predictor_cpu, "cpu", NN_weights_2, if_print = 1)
                    grad_store_1 = grad_store_1.to("cpu")
                    print("G_samples", G_samples)
                    print("grad_store_1",grad_store_1)
                    print("grad_store_2",grad_store_2)
                    print("grad_store_1_sum",grad_store_1.sum())
                    print("grad_store_2_sum",grad_store_2.sum())


                    wandb.log({"val_final_var_square_loss": var_square_loss})
                    #_, indices = torch.topk(NN_weights, model_cfg.batch_size_query) #select top k indices
                    #remaining index after top k values
                    #remaining_indices = list(set(list(range(pool_x.shape[0]))) - set(indices)) #needes to be checked

                    #add those to training
                    #train_x = torch.cat((train_x, pool_x[indices, ]), 0)
                    #train_y = torch.cat((train_y, pool_y[indices ]), 0)

                    #plot_visualization(pool_x[indices, ], pool_y[indices ], a,',pool')

                    #gp_model_track  = CustomizableGPModel(train_x, train_y, mean_module_track , base_kernel_track , likelihood_track ).to("cpu")
                    #gp_model_track.eval()
                    #likelihood_track.eval()
                    #model_predictor_dumi = ConstantValueNetwork(constant_value=0.0, output_size=1).to("cpu")
                    #model_predictor_dumi.eval()

                    #mean_track, loss_track = var_l2_loss_estimator(gp_model_track, test_x.to("cpu"), model_predictor_dumi, "cpu", n_samples_track)
                    #mean_actual = l2_loss(test_x, test_y, model_predictor, (test_x).device)
                    #wandb.log({"var_square_loss_track": loss_track.item(), "l2_loss_track": mean_track.item(), "l2_loss_actual_track": mean_actual.item()})



                    
                    #remove those points from pool 
                    #pool_x = pool_x[remaining_indices, ]
                    #pool_y = pool_y[remaining_indices]
                    #pool_sample_idx = pool_sample_idx[remaining_indices]
                    #direct_tensor_files = (train_x, train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx)

                    #plot_visualization(train_x, train_y,a)
                    #posterior_visualization(gp_model_track,test_x.to("cpu"),a)


                    cos_sim_pathwise = F.cosine_similarity(grad_store_1, grad_store_3, dim=0)
                    cos_sim_reinforce = F.cosine_similarity(grad_store_2, grad_store_3, dim=0)

                    mse_loss = torch.nn.MSELoss()
                    mse_pathwise = mse_loss(grad_store_1, grad_store_3)
                    mse_reinforce = mse_loss(grad_store_2, grad_store_3)

                    squared_true_tensor_pathwise = grad_store_1**2
                    mean_squared_tensor_pathwise = torch.mean(squared_true_tensor_pathwise)
                    pathwise_tensor_squared_mean_all.append(mean_squared_tensor_pathwise.item())

                    squared_true_tensor_reinforce = grad_store_2**2
                    mean_squared_tensor_reinforce = torch.mean(squared_true_tensor_reinforce)
                    reinforce_tensor_squared_mean_all.append(mean_squared_tensor_reinforce.item())

                    str_1 = "cos_sim_pathwise_"+str(G_samples)
                    str_2 = "cos_sim_reinforce_"+str(G_samples)
                    str_3 = "mse_pathwise_"+str(G_samples)
                    str_4 = "mse_reinforce_"+str(G_samples)
                    str_5 = "mean_squared_pathwise_"+str(G_samples)
                    str_6 = "mean_squared_reinforce_"+str(G_samples)

                    cos_sim_pathwise_all.append(cos_sim_pathwise.item())
                    cos_sim_reinforce_all.append(cos_sim_reinforce.item())
                    mse_pathwise_all.append(mse_pathwise.item())
                    mse_reinforce_all.append(mse_reinforce.item())
                    

                    top5_values, top5_indices = torch.topk(grad_store_1, 5)
                    least5_values, least5_indices = torch.topk(grad_store_1, 5, largest=False)
                    print("grad_store_1_top_5", top5_values)
                    print("grad_store_1_top_5_sum", top5_values.sum())
                    print("top_5_indices_grad_store_1", top5_indices)
                    print("grad_store_1_least_5", least5_values)
                    print("grad_store_1_least_5_sum", least5_values.sum())
                    print("least_5_indices_grad_store_1", least5_indices)

                    top5_values, top5_indices = torch.topk(grad_store_2, 5)
                    least5_values, least5_indices = torch.topk(grad_store_2, 5, largest=False)
                    print("grad_store_2_top_5", top5_values)
                    print("grad_store_2_top_5_sum", top5_values.sum())
                    print("top_5_indices_grad_store_2", top5_indices)
                    print("grad_store_2_least_5", least5_values)
                    print("grad_store_2_least_5_sum", least5_values.sum())
                    print("least_5_indices_grad_store_2", least5_indices)


                    wandb.log({str_1: cos_sim_pathwise.item(), str_2: cos_sim_reinforce.item(), str_3: mse_pathwise.item(), str_4: mse_reinforce.item(), str_5: mean_squared_tensor_pathwise.item(), str_6: mean_squared_tensor_reinforce.item()})

            
 
            cos_sim_pathwise_average = statistics.mean(cos_sim_pathwise_all)
            cos_sim_pathwise_variance = statistics.variance(cos_sim_pathwise_all)
            cos_sim_pathwise_standard_error = math.sqrt(cos_sim_pathwise_variance) / math.sqrt(len(cos_sim_pathwise_all))
            cos_sim_reinforce_average = statistics.mean(cos_sim_reinforce_all)
            cos_sim_reinforce_variance = statistics.variance(cos_sim_reinforce_all)
            cos_sim_reinforce_standard_error = math.sqrt(cos_sim_reinforce_variance) / math.sqrt(len(cos_sim_reinforce_all))

            mse_pathwise_average = statistics.mean(mse_pathwise_all)
            mse_pathwise_variance = statistics.variance(mse_pathwise_all)
            mse_pathwise_standard_error = math.sqrt(mse_pathwise_variance) / math.sqrt(len(mse_pathwise_all))
            mse_reinforce_average = statistics.mean(mse_reinforce_all)
            mse_reinforce_variance = statistics.variance(mse_reinforce_all)
            mse_reinforce_standard_error = math.sqrt(mse_reinforce_variance) / math.sqrt(len(mse_reinforce_all))


            pathwise_tensor_squared_mean_all_average = statistics.mean(pathwise_tensor_squared_mean_all)
            reinforce_tensor_squared_mean_all_average = statistics.mean(reinforce_tensor_squared_mean_all)


            print("cos_sim_pathwise_all",cos_sim_pathwise_all)
            print("cos_sim_reinforce_all",cos_sim_reinforce_all)
            print("mse_pathwise_all",mse_pathwise_all)
            print("mse_reinforce_all",mse_reinforce_all)

            wandb.log({str_1+"_average":cos_sim_pathwise_average, str_1+"_variance":cos_sim_pathwise_variance, str_1+"_standard_error":cos_sim_pathwise_standard_error})
            wandb.log({str_2+"_average":cos_sim_reinforce_average, str_2+"_variance":cos_sim_reinforce_variance, str_2+"_standard_error":cos_sim_reinforce_standard_error})
            wandb.log({str_3+"_average":mse_pathwise_average, str_3+"_variance":mse_pathwise_variance, str_3+"_standard_error":mse_pathwise_standard_error})
            wandb.log({str_4+"_average":mse_reinforce_average, str_4+"_variance":mse_reinforce_variance, str_4+"_standard_error":mse_reinforce_standard_error})
            wandb.log({"pathwise_tensor_squared_mean_all_average":pathwise_tensor_squared_mean_all_average, "reinforce_tensor_squared_mean_all_average": reinforce_tensor_squared_mean_all_average})






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command line arguments.")
    parser.add_argument("--config_file_path", type=str, help="Path to the JSON file containing the sweep configuration", default='config_sweep.json')
    parser.add_argument("--project_name", type=str, help="WandB project name", default='adaptive_sampling_gp')
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


    sweep_id = wandb.sweep(config_params, project=args.project_name, entity=ENTITY)
    # Run the agent
    wandb.agent(sweep_id, function=main_run_func)
    
   




