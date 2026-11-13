import argparse
import typing

import torch
import gpytorch
import higher
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
from torch.distributions import Categorical

import k_subset_sampling
from dataloader_enn import TabularDataset, TabularDatasetPool, TabularDatasetCsv, TabularDatasetPoolCsv
from enn import basenet_with_learnable_epinet_and_ensemble_prior
from variance_recall_enn import Recall_True, var_recall_estimator
from enn_loss_func import weighted_nll_loss

# Define a configuration class for dataset-related parameters
@dataclass
class DatasetConfig:
    def __init__(self, direct_tensors_bool: bool, csv_file_train=None, csv_file_test=None, csv_file_pool=None, y_column=None, shuffle=False):
        self.direct_tensors_bool = direct_tensors_bool
        self.csv_file_train = csv_file_train
        self.csv_file_test = csv_file_test
        self.csv_file_pool = csv_file_pool
        self.y_column = y_column      # Assuming same column name across above 3 sets
        self.shuffle = shuffle


@dataclass
class ModelConfig:
    access_to_true_pool_y: bool
    batch_size_query: int
    temp_k_subset: float
    meta_opt_lr: float
    meta_opt_weight_decay: float
    n_classes: int
    temp_recall: float


@dataclass
class TrainConfig:
    n_train_iter: int
    n_samples: int       # to calculate the variance
    G_samples: int
    n_iter_noise: int     # not used in regression but used in recall
    batch_size: int


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
    

"""# **Epistemic Neural Networks**"""
    
def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")    

def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, enn_config: ENNConfig, direct_tensor_files, Predictor, device, seed_training, if_print = 0):


    # Predictor here has already been pretrained


    # CHECK ALL THE DIMENSIONS of the tensors and datasets, and outputs from dataloaders


    # If train_x has dim [N,D]  and train_y has dim [N,1] then dataloader will give batch of train_x with dim [batch_size,D] and train_y with dim [batch_size,1]
    # If test_x has dim [N,D]  and test_y has dim [N] then dataloader will give batch of test_x with dim [batch_size,D] and test_y with dim [batch_size]
    # From the polyadic sampler we get [N,D] and [N] for x and y respectively   
    # Similarly, if they are in csv - then also the dimensions for x and y are [N,D] and [N] respectively

    # So in our case dataloader is returning y of dim [batch_size]

    # output of the neural network is usually [N,1] for regression and [N,C] for classification

    # ENN takes in x of dim [N,D] and z of dim [z_dim]
    # ENN outputs y of dim [N,n_classes]      (GP in gpytorch takes in inputs(while setting the train data) and outputs y of dim [N])
    # nn.CrossEntropyLoss()   -  only takes y_targets as [N]
    # nn.mse_loss() - will take in y_target depending on dimension of output of ENN - as it is [N,1] - therefore y_target should be [N,1]
    # currently ouputs from the GP is [N] and also the outputs from blr is [N]




    if dataset_config.direct_tensors_bool:
        assert direct_tensor_files != None, "direct_tensors_were_not_provided"
        
        init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, pool_sample_idx, test_sample_idx = direct_tensor_files
        
        
        dataset_train = TabularDataset(x = init_train_x, y = init_train_y)
        dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)
        
        # dataset_test = TabularDataset(x = test_x, y = test_y)
        # dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, shuffle=False)
        
        # dataset_pool = TabularDataset(x = pool_x, y = pool_y)
        # dataloader_pool = DataLoader(dataset_pool, batch_size=pool_x.size(0), shuffle=False)
        
        # x_combined = torch.cat([init_train_x, pool_x], dim=0)
        # y_combined = torch.cat([init_train_y, pool_y], dim=0)
        # dataset_train_and_pool = TabularDatasetPool(x=x_combined, y=y_combined)
        # dataloader_train_and_pool = DataLoader(dataset_train_and_pool, batch_size=model_config.batch_size, shuffle=dataset_config.shuffle)
        
        
        

    
    
    
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


        
        dataset_train = TabularDataset(x = init_train_x, y = init_train_y)
        dataloader_train = DataLoader(dataset_train, batch_size=model_config.batch_size, shuffle=dataset_config.shuffle)
        
        #dataset_test = TabularDatasetCsv(device, csv_file=dataset_config.csv_file_test, y_column=dataset_config.y_column)
        #dataloader_test = DataLoader(dataset_test, batch_size=model_config.batch_size, shuffle=False)       # gives batch for test features and label    (both in float 32)

        #dataset_pool = TabularDatasetCsv(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
        #pool_size = len(dataset_pool)
        #dataloader_pool = DataLoader(dataset_pool, batch_size=pool_size, shuffle=False)       # gives all the pool features and label   (both in float 32) - needed for input in NN_weights

        #dataset_train_and_pool = TabularDatasetPoolCsv(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
        #dataloader_train_and_pool = DataLoader(dataset_pool_train, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)       # gives batch of the pool features and label   (both in float 32) - needed for updating the posterior of ENN - as we will do batchwise update
        
        # dataset_train = TabularDataset(x = init_train_x, y = init_train_y)
        # dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)
        
        # dataset_test = TabularDataset(x = test_x, y = test_y)
        # dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, shuffle=False)

        # dataset_pool = TabularDataset(x = pool_x, y = pool_y)
        # dataloader_pool = DataLoader(dataset_pool, batch_size=pool_x.size(0), shuffle=False)
        
        # x_combined = torch.cat([init_train_x, pool_x], dim=0)
        # y_combined = torch.cat([init_train_y, pool_y], dim=0)
        # dataset_train_and_pool = TabularDatasetPool(x=x_combined, y=y_combined)
        # dataloader_train_and_pool = DataLoader(dataset_train_and_pool, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)


        pool_sample_idx = None 
        test_sample_idx = None
        
    
    pool_size = pool_x.size(0)
    sample, label = dataset_train[0]
    input_feature_size = sample.shape[0]
    




    #input_feature_size = init_train_x.size(1)
    #NN_weights = NN_feature_weights(input_feature_size, model_config.hidden_sizes_weight_NN, 1).to(device)
    #meta_opt = optim.Adam(NN_weights.parameters(), lr=model_config.meta_opt_lr, weight_decay=model_config.meta_opt_weight_decay)


    NN_weights = torch.full([pool_size], math.log(1.0 / pool_size), requires_grad=True, device=device)
    meta_opt = optim.Adam([NN_weights], lr=model_config.meta_opt_lr, weight_decay=model_config.meta_opt_weight_decay)
    SubsetOperatorthis = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, False).to(device)
    SubsetOperatortestthis = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, True).to(device)

    
    ENN_model = basenet_with_learnable_epinet_and_ensemble_prior(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.exposed_layers, enn_config.z_dim, enn_config.learnable_epinet_hiddens, enn_config.hidden_sizes_prior, enn_config.seed_base, enn_config.seed_learnable_epinet, enn_config.seed_prior_epinet, enn_config.alpha).to(device)

    # Need to do this because ENN_model itself has some seeds and we need to set the seed for the whole training process here
    torch.manual_seed(seed_training)
    np.random.seed(seed_training)
    if device=="cuda":
        torch.cuda.manual_seed(seed_training) # Sets the seed for the current GPU
        torch.cuda.manual_seed_all(seed_training) # Sets the seed for all GPUs
    
        

    loss_fn_init = nn.CrossEntropyLoss()
    optimizer_init = optim.Adam(ENN_model.parameters(), lr=enn_config.ENN_opt_lr, weight_decay=enn_config.ENN_opt_weight_decay)
    enn_loss_list = []
    for i in range(enn_config.n_ENN_iter):
        ENN_model.train()
        for (inputs, labels) in dataloader_train:   #check what is dim of inputs, labels, ENN_model(inputs,z)
            aeverage_loss = 0
            optimizer_init.zero_grad()
            for j in range(enn_config.z_samples): 
                z = torch.randn(enn_config.z_dim, device=device)
                
                outputs = ENN_model(inputs,z)
                
                #print("outputs:", outputs)
                #print("labels:", labels)
                #labels = torch.tensor(labels, dtype=torch.long, device=device)
                
                loss = loss_fn_init(outputs, labels.squeeze().long())/enn_config.z_samples
                loss.backward()
                aeverage_loss += loss

            optimizer_init.step()
            
            enn_loss_list.append(float(aeverage_loss.detach().to('cpu').numpy()))
     
    #samples_list=torch.empty((0), dtype=torch.float32, device=device)
    prediction_probs_pos_list=torch.empty((0), dtype=torch.float32, device=device)
    optimizer_init.zero_grad()
     
    for i in range(train_config.n_samples):
        z_test = torch.randn(enn_config.z_dim, device=device)
        prediction = ENN_model(test_x, z_test) #x is all data
        prediction_probs =  torch.softmax(prediction, dim=1)
        prediction_probs_pos = (prediction_probs[:,-1:].squeeze()).unsqueeze(dim=0)
        print(str(i)+"_prediction_prob_pos:",prediction_probs_pos)
        #distribution = Categorical(logits=prediction)
        #samples = distribution.sample((1,))
        #samples_list = torch.cat((samples_list,samples),0)
        prediction_probs_pos_list = torch.cat((prediction_probs_pos_list,prediction_probs_pos),0)
      
    #posterior_mean = torch.mean(samples_list, dim=0)
    #posterior_std = torch.std(samples_list, dim=0)
    posterior_mean = torch.mean(prediction_probs_pos_list, dim=0)
    posterior_std = torch.std(prediction_probs_pos_list, dim=0)

    
    dataset_test = TabularDataset(x = test_x, y = test_y)
    dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, shuffle=False)
    #dataloader test must have shuffle false


    meta_mean, meta_loss = var_recall_estimator(ENN_model, dataloader_test, Predictor, device, model_config.temp_recall, enn_config.z_dim, train_config.n_samples, train_config.n_iter_noise)

    recall_actual = Recall_True(dataloader_test, Predictor, device)
    wandb.log({"meta_loss_initial": meta_loss.item(), "meta_mean_intial": meta_mean.item(), "recall_actual_initial": recall_actual.item()})


    fig_enn_training = plt.figure()
    plt.plot(list(range(len(enn_loss_list))),enn_loss_list)
    plt.title('ENN initial training loss')
    plt.legend()
    wandb.log({'ENN initial training loss': wandb.Image(fig_enn_training)})
    plt.close(fig_enn_training)
    
    if init_train_x.size(1) == 1:
        fig_enn_posterior = plt.figure()
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy())
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()-2*posterior_std.detach().cpu().numpy(),alpha=0.2)
        plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()+2*posterior_std.detach().cpu().numpy(),alpha=0.2)
        wandb.log({'ENN initial posterior': wandb.Image(fig_enn_posterior)})
        plt.close(fig_enn_posterior)



    train(ENN_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatorthis, Predictor, pool_sample_idx, if_print = if_print)
    var_recall = test(ENN_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatortestthis, Predictor, pool_sample_idx, if_print = if_print)
    
    return var_recall

def train(ENN_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatorthis, Predictor, pool_sample_idx, if_print = 0):
  print("NN_weights_in_start:", NN_weights)
  
  dataset_train = TabularDataset(x = init_train_x, y = init_train_y)
  dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)
    
  dataset_test = TabularDataset(x = test_x, y = test_y)
  dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, shuffle=False)
  #dataloader test must have shuffle false

  dataset_pool = TabularDataset(x = pool_x, y = pool_y)
  dataloader_pool = DataLoader(dataset_pool, batch_size=pool_x.size(0), shuffle=False)
    
  x_combined = torch.cat([init_train_x, pool_x], dim=0)
  y_combined = torch.cat([init_train_y, pool_y], dim=0)
  dataset_train_and_pool = TabularDatasetPool(x=x_combined, y=y_combined)
  dataloader_train_and_pool = DataLoader(dataset_train_and_pool, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)


  ENN_model.train()

  for i in range(train_config.n_train_iter):    # Should we do this multiple times or not
    start_time = time.time()

    meta_opt.zero_grad()
    aeverage_meta_loss = 0.0

    #pool_weights = NN_weights(pool_x)   #pool_weights has shape [pool_size,1]
    #pool_weights_t = pool_weights.t()  #convert pool_weights to shape 
    #soft_k_vector = SubsetOperator(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]

    if model_config.access_to_true_pool_y:
        dataset_train_and_pool = dataset_train_and_pool     
    else:
        z_pool_dumi = torch.randn(enn_config.z_dim, device=device)
        pool_logits_dumi = ENN_model(pool_x, z_pool_dumi).squeeze()    # assuming this can be handled by the GPUs otherwise put it in batches
        distribution = Categorical(logits=pool_logits_dumi)
        pool_y_dumi = distribution.sample((1,)).squeeze()
        y_enn = torch.cat([init_train_y,pool_y_dumi], dim=0)
        dataset_train_and_pool.update_targets(y_enn)

    for g in range(train_config.G_samples):
        intermediate_time_1 = time.time()
       

        NN_weights_unsqueezed = NN_weights.unsqueeze(0)       #[1, pool_size]
        soft_k_vector = SubsetOperatorthis(NN_weights_unsqueezed)  #soft_k_vector has shape  [1,pool_size]
        soft_k_vector_squeeze = soft_k_vector.squeeze()  #soft_k_vector_squeeze has shape  [pool_size]
        clipped_soft_k_vector_squeeze = torch.clamp(soft_k_vector_squeeze, min=-float('inf'), max=1.0)


        #input_feature_size = init_train_x.size(1)
        init_train_size = init_train_x.size(0)

        w_train = torch.ones(init_train_size, requires_grad = True, device=device)
        w_enn = torch.cat([w_train,clipped_soft_k_vector_squeeze])


        ENN_opt = torch.optim.Adam(ENN_model.parameters(), lr=enn_config.ENN_opt_lr, weight_decay=enn_config.ENN_opt_weight_decay)
        
        with higher.innerloop_ctx(ENN_model, ENN_opt, copy_initial_weights=False) as (fnet, diffopt):
            fnet_loss_list = []
            for j in range(enn_config.n_ENN_iter):
                for (idx_batch, x_batch, label_batch) in dataloader_train_and_pool:
                    aeverage_loss = 0.0
                    for k in range(enn_config.z_samples):
                        z = torch.randn(enn_config.z_dim, device=device)
                        fnet_logits = fnet(x_batch,z)
                        batch_log_probs = F.log_softmax(fnet_logits, dim=1)
                        weights_batch = w_enn[idx_batch]
                        ENN_loss = weighted_nll_loss(batch_log_probs, label_batch.squeeze().long(), weights_batch)/enn_config.z_samples
                        aeverage_loss += ENN_loss
                    diffopt.step(aeverage_loss)      ## Need to find a way where we can accumulate the gradients and then take the diffopt.step()
                    fnet_loss_list.append(float(aeverage_loss.detach().to('cpu').numpy()))
            intermediate_time_2 = time.time()
            meta_mean, meta_loss = var_recall_estimator(fnet, dataloader_test, Predictor, device, model_config.temp_recall, enn_config.z_dim, train_config.n_samples, train_config.n_iter_noise)
            meta_loss = meta_loss/train_config.G_samples
            meta_loss.backward()
            aeverage_meta_loss += meta_loss
            wandb.log({"epoch+g_samples": i+g, "time_taken_per_g":intermediate_time_2-intermediate_time_1, "meta_loss": meta_loss.item(), "meta_mean": meta_mean.item()})


            # ideally we should aeverage over meta mean as well but we are not doing it right now


    intermediate_time_3 = time.time()
    meta_opt.step()

    recall_actual = Recall_True(dataloader_test, Predictor, device)

    
    if i <=50  or i >= train_config.n_train_iter-2: #only plot first few
        #samples_list=torch.empty((0), dtype=torch.float32, device=device)
        prediction_probs_pos_list=torch.empty((0), dtype=torch.float32, device=device)
     
        for q in range(train_config.n_samples):
            z_test = torch.randn(enn_config.z_dim, device=device)
            prediction = fnet(test_x, z_test) #x is all data
            prediction_probs =  torch.softmax(prediction, dim=1)
            prediction_probs_pos = (prediction_probs[:,-1:].squeeze()).unsqueeze(dim=0)
            #distribution = Categorical(logits=prediction)
            #samples = distribution.sample((1,))
            #samples_list = torch.cat((samples_list,samples),0)
            prediction_probs_pos_list = torch.cat((prediction_probs_pos_list,prediction_probs_pos),0)
      
        #posterior_mean = torch.mean(samples_list, dim=0)
        #posterior_std = torch.std(samples_list, dim=0)
        posterior_mean = torch.mean(prediction_probs_pos_list, dim=0)
        posterior_std = torch.std(prediction_probs_pos_list, dim=0)

        fig_fnet_training = plt.figure()
        plt.plot(list(range(len(fnet_loss_list))),fnet_loss_list)
        plt.title('fnet loss within training at training iter ' + str(i))
        plt.legend()
        wandb.log({'Fnet training loss'+ str(i): wandb.Image(fig_fnet_training)})
        plt.close(fig_fnet_training)

        if init_train_x.size(1) == 1:

            fig_fnet_posterior = plt.figure()
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy())
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()-2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            plt.scatter(test_x.squeeze().cpu().numpy(),posterior_mean.detach().cpu().numpy()+2*posterior_std.detach().cpu().numpy(),alpha=0.2)
            plt.title('fnet posterior within training at training iter ' + str(i))
            wandb.log({'Fnet posterior'+ str(i): wandb.Image(fig_fnet_posterior)})
            plt.close(fig_fnet_posterior)




    # _, indices = torch.topk(NN_weights, model_config.batch_size_query)
    # hard_k_vector = torch.zeros_like(NN_weights)
    # hard_k_vector[indices] = 1.0
    # # chane the y_enn here and dataloaders
    # #Train the copy of ENN_model -> ENN_model_new here so that weights at the start of the loop are same
    # var_square_loss_hard = var_l2_loss_estimator(ENN_model_new, test_x, Predictor, device, None)
   
    if pool_sample_idx != None:
        #NN_weights_values, NN_weights_indices = torch.topk(NN_weights, model_config.batch_size_query)
        #selected_clusters_from_pool = pool_sample_idx[NN_weights_indices]
        #selected_points_indices = {f"selected_point_{j}_indices": NN_weights_indices[j].item() for j in range(model_config.batch_size_query)}
        #selected_clusters_from_pool_tensor_data = {f"selected_point_{j}_belongs_to_the_cluster": selected_clusters_from_pool[j].item() for j in range(model_config.batch_size_query)}
        #wandb.log({"epoch": i, "var_square_loss": average_meta_loss.item(), "var_square_loss_hard":var_square_loss_hard.item(),"mean_square_loss": mean_square_loss.item(), "l_2_loss_actual":l_2_loss_actual.item(),**selected_points_indices,**selected_clusters_from_pool_tensor_data})
        weights_dict = {f"weight_{a}": NN_weights[a].detach().cpu().item() for a in range(NN_weights.size(0))}
        wandb.log({"epoch": i, "time_taken_per_epoch":intermediate_time_3-start_time, "aeverage_var_recall": aeverage_meta_loss.item(), "mean_recall": meta_mean.item(), "recall_actual":recall_actual.item(), **weights_dict})
        #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
    else:
        weights_dict = {f"weights/weight_{a}": weight.detach().cpu().item() for a, weight in enumerate(NN_weights)}
        wandb.log({"epoch": i,  "time_taken_per_epoch":intermediate_time_3-start_time, "var_recall": aeverage_meta_loss.item(), "mean_recall": meta_mean.item(), "recall_actual":recall_actual.item(), **weights_dict})
        #wandb.log({"weights": [weight.detach().cpu().item() for weight in NN_weights]})
        
        #wandb.log(weights_dict)
    
    


def test(ENN_model, init_train_x, init_train_y, pool_x, pool_y, test_x, test_y, device, dataset_config, model_config, train_config, enn_config, NN_weights, meta_opt, SubsetOperatortestthis, Predictor, pool_sample_idx, if_print = 0):
    

    x_combined = torch.cat([init_train_x, pool_x], dim=0)
    y_combined = torch.cat([init_train_y, pool_y], dim=0)
    dataset_train_and_pool = TabularDatasetPool(x=x_combined, y=y_combined)
    dataloader_train_and_pool = DataLoader(dataset_train_and_pool, batch_size=train_config.batch_size, shuffle=dataset_config.shuffle)
    

    _, indices = torch.topk(NN_weights, model_config.batch_size_query)
    hard_k_vector = torch.zeros_like(NN_weights)
    hard_k_vector[indices] = 1.0
    init_train_size = init_train_x.size(0)
    w_train = torch.ones(init_train_size, requires_grad = True).to(device)
    w_enn = torch.cat([w_train,hard_k_vector])
    
    ENN_model.train()
    
    
    ENN_opt = torch.optim.Adam(ENN_model.parameters(), lr=enn_config.ENN_opt_lr, weight_decay=enn_config.ENN_opt_weight_decay)


    for i in range(enn_config.n_ENN_iter):
        for (idx_batch, x_batch, label_batch) in dataloader_train_and_pool:
            aeverage_loss = 0.0
            ENN_opt.zero_grad()
            for j in range(enn_config.z_samples):
                z = torch.randn(enn_config.z_dim, device=device)
                fnet_logits = ENN_model(x_batch,z)
                batch_log_probs = F.log_softmax(fnet_logits, dim=1)
                weights_batch = w_enn[idx_batch]
                ENN_loss = weighted_nll_loss(batch_log_probs, label_batch.squeeze().long(), weights_batch)/enn_config.z_samples
                ENN_loss.backward()
                aeverage_loss += ENN_loss
            ENN_opt.step() 

    # Assuming that whole of test_x can be processed at once by both ENN and predictor - and doesnot need a dataloader - can change it later
    dataset_test = TabularDataset(x = test_x, y = test_y)
    dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, shuffle=False)
  
    meta_mean, meta_loss = var_recall_estimator(ENN_model, dataloader_test, Predictor, device, model_config.temp_recall, enn_config.z_dim, train_config.n_samples, train_config.n_iter_noise)
    recall_actual = Recall_True(dataloader_test, Predictor, device)

    #print("l_2_loss_actual:", l_2_loss_actual)
    
    
    
    if pool_sample_idx != None:
        #NN_weights_values, NN_weights_indices = torch.topk(NN_weights, model_config.batch_size_query)
        #selected_clusters_from_pool = pool_sample_idx[NN_weights_indices]
        #selected_points_indices = {f"val_selected_point_{j}_indices": NN_weights_indices[j].item() for j in range(model_config.batch_size_query)}
        #selected_clusters_from_pool_tensor_data = {f"val_selected_point_{j}_belongs_to_the_cluster": selected_clusters_from_pool[j].item() for j in range(model_config.batch_size_query)}
        #wandb.log({"val_var_square_loss": var_square_loss.item(), "val_mean_square_loss": mean_square_loss.item(), "val_l_2_loss_actual":l_2_loss_actual.item(), **selected_points_indices, **selected_clusters_from_pool_tensor_data})
        wandb.log({"val_var_recall": meta_loss.item(), "val_mean_recall": meta_mean.item(), "val_recall_actual":recall_actual.item()})
        
    else:
        wandb.log({"val_var_recall": meta_loss.item(), "val_mean_recall": meta_mean.item(), "val_recall_actual":recall_actual.item()})
    
    print("NN_weights_in_end:", NN_weights)
    
    return meta_loss