import argparse
import typing

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as distributions
import numpy as np
from dataclasses import dataclass
import time
import higher
from datetime import datetime
import matplotlib.pyplot as plt

from dataloader import TabularDataset
from dataloader import TabularDatasetPool

import k_subset_sampling #import SubsetOperator
from nn_feature_weights import NN_feature_weights
from ENN import basenet_with_learnable_epinet_and_ensemble_prior


from enn_loss_func import weighted_nll_loss


from var_recall_estimator import *     

# Define a configuration class for dataset-related parameters
@dataclass
class DatasetConfig:
    csv_file_train: str
    csv_file_test: str
    csv_file_pool: str
    y_column: str  # Assuming same column name across above 3 sets


@dataclass
class ModelConfig:
    batch_size_train: int
    batch_size_test: int
    batch_size_query: int
    temp_k_subset: float
    hidden_sizes_weight_NN: list
    meta_opt_lr: float
    n_classes: int
    n_epoch: int
    init_train_lr: float
    init_train_weight_decay: float
    n_train_init: int
    meta_opt_weight_decay: float
    




@dataclass
class TrainConfig:
    n_train_iter: int
    n_ENN_iter: int
    ENN_opt_lr: float
    ENN_opt_weight_decay: float
    temp_var_recall: float
    z_dim: int
    N_iter: int
    seed_var_recall: int
    N_iter_var_recall_est: int


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

 
def train(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device, NN_weights, meta_opt, SubsetOperator, ENN, Predictor, if_print):
  ENN.train()
  meta_loss_list = []
  for i in range(train_config.n_train_iter):    # Should we do this multiple times or not
    start_time = time.time()
    x_pool, y_pool = next(iter(dataloader_pool))
    #x_pool = x_pool.to(device)
    #y_pool = y_pool.to(device)
    pool_weights = NN_weights(x_pool)   #pool_weights has shape [pool_size,1]
    pool_weights_t = pool_weights.t()  #convert pool_weights to shape [1, pool_size]

    #set seed
    soft_k_vector = SubsetOperator(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]
    soft_k_vector_squeeze = soft_k_vector.squeeze()


    z_pool = torch.randn(train_config.z_dim, device=device) # set seed for z #set to device
    x_pool_label_ENN_logits = ENN(x_pool,z_pool)  #use here complete dataset
    x_pool_label_ENN_probabilities = F.softmax(x_pool_label_ENN_logits, dim=1) #see if dim=1 is correct
    x_pool_label_ENN_categorical = distributions.Categorical(x_pool_label_ENN_probabilities)
    x_pool_label_ENN = x_pool_label_ENN_categorical.sample() # set seed for labels           # Do we need to take aeverages over multiple z's here? - No, do this for multiple z's



    ENN_opt = torch.optim.Adam(ENN.parameters(), lr=train_config.ENN_opt_lr, weight_decay=train_config.ENN_opt_weight_decay)
    #print('ENN model weights',ENN.learnable_epinet_layers[0].weight)
                                                                              #copy_initial_weights - will be important if we are doing multisteps  # how to give initial training weights to ENN -- this is resolved , if we use same instance of the model everywhere - weights get stored
    meta_opt.zero_grad()

    enn_loss_list = []
    with higher.innerloop_ctx(ENN, ENN_opt, copy_initial_weights=False) as (fnet, diffopt):
      for _ in range(train_config.n_ENN_iter):

        for (idx_batch, x_batch, label_batch) in dataloader_pool_train:

          #idx_batch = idx_batch.to(device)
          #x_batch = x_batch.to(device)
          #label_batch =  label_batch.to(device)
          z_pool_train = torch.randn(train_config.z_dim, device=device)

          fnet_logits = fnet(x_batch, z_pool_train)    # Forward pass (outputs are logits) #DEFINE fnet sampler through fnet
          batch_log_probs = F.log_softmax(fnet_logits, dim=1)     #see if here dim=1 is correct or not   # Apply log-softmax to get log probabilities


          batch_weights = soft_k_vector_squeeze[idx_batch]        # Retrieve weights for the current batch
          x_batch_label_ENN = x_pool_label_ENN[idx_batch]         # Retrieve labels for the current batch
          #print('batch_weights:', batch_weights)
          #print('x_batch_label_ENN:', x_batch_label_ENN)
          # Calculate loss
          ENN_loss = weighted_nll_loss(batch_log_probs,x_batch_label_ENN,batch_weights)       #expects log_probabilities as inputs    #CHECK WORKING OF THIS
          if if_print == 1:
            print("ENN_loss:", ENN_loss)
          #print("ENN_loss:",ENN_loss)
          diffopt.step(ENN_loss)
          #print('ENN model weights inside training loop',fnet.learnable_epinet_layers[0].weight)
        
        enn_loss_list.append(float(ENN_loss.detach().to('cpu').numpy()))

      #derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters
      meta_loss = var_recall_estimator(fnet, dataloader_test, Predictor, device, para = {'tau': train_config.temp_var_recall, 'z_dim': train_config.z_dim, 'N_iter': train_config.N_iter ,'if_print':if_print, 'seed_var_recall':train_config.seed_var_recall, 'N_iter_var_recall_est':train_config.N_iter_var_recall_est})     #see where does this calculation for meta_loss happens that is it outside the innerloop_ctx or within it
      if if_print == 1:
        print("meta_loss:", meta_loss)
      meta_loss.backward()
      recall_true = Recall_True(dataloader_test, Predictor, device)
      if if_print == 1:
        print("recall_true:", recall_true)
      

      if i <= 0 and i >= train_config.n_train_iter-2: #only plot first few
        plt.plot(list(range(len(enn_loss_list))),enn_loss_list)
        plt.title('ENN loss within training at training iter ' + str(i))
        plt.show()
    meta_opt.step()
    meta_loss_print = meta_loss.detach().to('cpu')
    meta_loss_list.append(float(meta_loss_print.numpy()))
    



  if if_print == 1:
    print('meta_loss_list', meta_loss_list)
  plt.plot(list(range(len(meta_loss_list))),meta_loss_list)
  plt.title('meta_loss vs training iter')
  plt.show()


    # log all important metrics and also save model configs

def test(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device,  NN_weights, SubsetOperatortest, ENN, Predictor, if_print):

  ENN.train()
  x_pool,y_pool = next(iter(dataloader_pool))                     #corect this with arguments if we needed
  #x_pool = x_pool.to(device)
  #y_pool = y_pool.to(device)
  pool_weights = NN_weights(x_pool)   #pool_weights has shape [pool_size,1]
  pool_weights_t = pool_weights.t()  #convert pool_weights to shape [1, pool_size]

  #set seed
  hard_k_vector = SubsetOperatortest(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]
  hard_k_vector_squeeze = hard_k_vector.squeeze()


  ENN_opt = torch.optim.Adam(ENN.parameters(), lr=train_config.ENN_opt_lr, weight_decay=train_config.ENN_opt_weight_decay)

  
  with higher.innerloop_ctx(ENN, ENN_opt, track_higher_grads=False) as (fnet, diffopt):

    for _ in range(train_config.n_ENN_iter):

      for (idx_batch, x_batch, label_batch) in dataloader_pool_train:

          
          #idx_batch = idx_batch.to(device)
          #x_batch = x_batch.to(device)
          #label_batch =  label_batch.to(device)
          z_pool_train = torch.randn(train_config.z_dim, device=device)

          fnet_logits = fnet(x_batch, z_pool_train)    # Forward pass (outputs are logits) #DEFINE fnet sampler through fnet
          batch_log_probs = F.log_softmax(fnet_logits, dim=1)     #see if here dim=1 is correct or not   # Apply log-softmax to get log probabilities


          batch_weights = hard_k_vector_squeeze[idx_batch]        # Retrieve weights for the current batch
          y_batch = y_pool[idx_batch]         # Retrieve labels for the current batch

          y_batch = torch.tensor(y_batch, dtype=torch.long)
          y_batch = torch.squeeze(y_batch)
          # Calculate loss
          ENN_loss = weighted_nll_loss(batch_log_probs,y_batch,batch_weights)       #expects log_probabilities as inputs    #CHECK WORKING OF THIS
          if if_print == 1:
            print("ENN_loss:",ENN_loss)
          diffopt.step(ENN_loss)

    meta_loss = var_recall_estimator(fnet, dataloader_test, Predictor, device, para = {'tau': train_config.temp_var_recall, 'z_dim': train_config.z_dim, 'N_iter': train_config.N_iter ,'if_print':if_print, 'seed_var_recall':train_config.seed_var_recall, 'N_iter_var_recall_est':train_config.N_iter_var_recall_est})     #see where does this calculation for meta_loss happens that is it outside the innerloop_ctx or within it
    recall_true = Recall_True(dataloader_test, Predictor, device)
    if if_print == 1:
      print("meta_loss:", meta_loss)
      print("recall_true:", recall_true)
    #see what does detach() do and if needed here


  #log and print important things here

def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, enn_config: ENNConfig, Predictor, device, if_print = 0):


    # Predictor here has already been pretrained


    # ------ see how to define a global seed --------- and separate controllable seeds for reproducibility
    #torch.manual_seed(40) 
    # see how to do this for dataset_train and dataset_test



    #to device and seed for this ----
    dataset_train = TabularDataset(device, csv_file=dataset_config.csv_file_train, y_column=dataset_config.y_column)
    dataloader_train = DataLoader(dataset_train, batch_size=model_config.batch_size_train, shuffle=True)     # gives batch for training features and labels  (both in float 32)

    dataset_test = TabularDataset(device, csv_file=dataset_config.csv_file_test, y_column=dataset_config.y_column)
    dataloader_test = DataLoader(dataset_test, batch_size=model_config.batch_size_test, shuffle=True)       # gives batch for test features and label    (both in float 32)

    dataset_pool = TabularDataset(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    pool_size = len(dataset_pool)
    dataloader_pool = DataLoader(dataset_pool, batch_size=pool_size, shuffle=False)       # gives all the pool features and label   (both in float 32) - needed for input in NN_weights

    dataset_pool_train = TabularDatasetPool(device, csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    dataloader_pool_train = DataLoader(dataset_pool_train, batch_size=model_config.batch_size_train, shuffle=True)       # gives batch of the pool features and label   (both in float 32) - needed for updating the posterior of ENN - as we will do batchwise update


    all_data_list = [dataset_train, dataset_pool, dataset_test]
    all_data_name = ['train','pool','test']

    sample, label = dataset_train[0]
    input_feature_size = sample.shape[0]       # Size of input features  ---- assuming 1D features

    NN_weights = NN_feature_weights(input_feature_size, model_config.hidden_sizes_weight_NN, 1).to(device)
    # --- TO INITIAL PARAMETRIZATION WITHIN  [0,1] , ALSO SET SEED ----------


    meta_opt = optim.Adam(NN_weights.parameters(), lr=model_config.meta_opt_lr, weight_decay=model_config.meta_opt_weight_decay)       # meta_opt is optimizer for parameters of NN_weights

    #seed for this
    SubsetOperator = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, False).to(device)

    #seed for this
    SubsetOperatortest = k_subset_sampling.SubsetOperator(model_config.batch_size_query, device, model_config.temp_k_subset, True).to(device)


    # to_device
    ENN = basenet_with_learnable_epinet_and_ensemble_prior(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.exposed_layers, enn_config.z_dim, enn_config.learnable_epinet_hiddens, enn_config.hidden_sizes_prior, enn_config.seed_base, enn_config.seed_learnable_epinet, enn_config.seed_prior_epinet, enn_config.alpha).to(device)


    loss_fn_init = nn.CrossEntropyLoss()
    optimizer_init = optim.Adam(ENN.parameters(), lr=model_config.init_train_lr, weight_decay=model_config.init_train_weight_decay)
    # ------- seed for this training
    # ------- train ENN on initial training data  # save the state - ENN_initial_state  # define a separate optimizer for this # how to sample z's ---- separately for each batch
    # ------- they also sampled the data each time and not a dataloader - kind of a bootstrap
    #print('ENN model weights',ENN.learnable_epinet_layers[0].weight)
    enn_loss_list = []
    for i in range(model_config.n_train_init):
        ENN.train()
        for (inputs, labels) in dataloader_train:
            #inputs, labels =  inputs.to(device), labels.to(device)
            z = torch.randn(enn_config.z_dim, device=device)   #set seed for this  #set to_device for this
            optimizer_init.zero_grad()
            outputs = ENN(inputs,z)

            labels = torch.tensor(labels, dtype=torch.long, device=device)
            loss = loss_fn_init(outputs, torch.squeeze(labels))
            if if_print == 1:
              print("ENN_init_loss:",loss)
            loss.backward()
            optimizer_init.step()

        enn_loss_list.append(float(loss.detach().to('cpu').numpy()))  
    plt.plot(list(range(len(enn_loss_list))),enn_loss_list)
    plt.title('ENN initial loss vs training iter')
    plt.show()

    #initial ENN training completed, now print loss on 3 datasets
    for i in range(3):
      List_all = []
      inputs = all_data_list[i].x 
      labels = all_data_list[i].y
      labels = torch.tensor(labels, dtype=torch.long)
      for j in range(train_config.N_iter):
        z = torch.randn(enn_config.z_dim, device=device)
      
        outputs = ENN(inputs,z)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        List_all.append(outputs)
      mean_outputs = torch.mean(torch.stack(List_all),0)
      prediction = mean_outputs[:,1]
      prediction = (prediction >= 0.35) #smaller threshold
      prediction = torch.tensor(prediction, dtype=torch.long)
      #prediction = torch.argmax(mean_outputs,1)
      loss = loss_fn_init(mean_outputs, torch.squeeze(labels))
      accuracy = torch.mean(torch.eq(prediction,labels).double())

      x = torch.sum(torch.mul(labels, prediction))
      y = torch.sum(labels)
      recall = x/y
      #print(recall, mean_outputs, prediction, labels)
      print('initial ENN_data',all_data_name[i],'cross entropy loss',float(loss),'accuracy',float(accuracy),'recall',float(recall))
 

    # Predictor =       # model for which we will evaluate recall   # load pretrained weights for the Predictor or train it


    #var should use dataset_pool?
    #should return something?
    t1 = datetime.now()

    for epoch in range(model_config.n_epoch):
      train(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device, NN_weights, meta_opt, SubsetOperator, ENN, Predictor, if_print = if_print)
      if if_print >= 0:
        t2 = datetime.now()
        delta = (t2-t1).total_seconds()/60
        t1 = datetime.now()
        print('training epoch ends in ', round(delta,2), 'minutes.') 

    print('test starts')
    test(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device,  NN_weights, SubsetOperatortest, ENN, Predictor, if_print = if_print)
