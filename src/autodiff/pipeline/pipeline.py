import argparse
import typing

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as distributions

from dataclasses import dataclass

import higher

from dataloader import TabularDataset
from dataloader import TabularDatasetPool

from k_subset_sampling import SubsetOperator
from NN_feature_weights import NN_feature_weights
from enn import basenet_with_learnable_epinet_and_ensemble_prior


from ENN_loss_func import weighted_nll_loss


from var_recall_estimator import var_recall_estimator    #Yuanzhe

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




@dataclass
class TrainConfig:
    n_train_iter: int
    n_ENN_iter: int
    ENN_opt_lr: float


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

def experiment(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, enn_config: ENNConfig, Predictor):


    # Predictor here has already been pretrained


    # ------ see how to define a global seed --------- and separate controllable seeds for reproducibility
    # see how to do this for dataset_train and dataset_test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ---------- ADD TO DEVICE ---------- everywhere, wherever necessaary


    #to device and seed for this ----
    dataset_train = TabularDataset(csv_file=dataset_config.csv_file_train, y_column=dataset_config.y_column)
    dataset_train = TabularDataset(csv_file=csv_file_train, y_column=y_column)
    dataloader_train = DataLoader(dataset_train, batch_size=model_config.batch_size_train, shuffle=True)     # gives batch for training features and labels  (both in float 32)

    dataset_test = TabularDataset(csv_file=dataset_config.csv_file_test, y_column=dataset_config.y_column)
    dataloader_test = DataLoader(dataset_test, batch_size=model_config.batch_size_test, shuffle=True)       # gives batch for test features and label    (both in float 32)

    dataset_pool = TabularDataset(csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    pool_size = len(dataset_pool)
    dataloader_pool = DataLoader(dataset_pool, batch_size=pool_size, shuffle=False)       # gives all the pool features and label   (both in float 32) - needed for input in NN_weights

    dataset_pool_train = TabularDatasetPool(csv_file=dataset_config.csv_file_pool, y_column=dataset_config.y_column)
    dataloader_pool_train = DataLoader(dataset_pool_train, batch_size=model_config.batch_size_train, shuffle=True)       # gives batch of the pool features and label   (both in float 32) - needed for updating the posterior of ENN - as we will do batchwise update



    sample, label = dataset_train[0]
    input_feature_size = sample.shape[0]       # Size of input features  ---- assuming 1D features

    NN_weights = NN_feature_weights(input_feature_size, model_config.hidden_sizes_weight_NN, 1)
    # --- TO INITIAL PARAMETRIZATION WITHIN  [0,1] , ALSO SET SEED ----------


    meta_opt = optim.Adam(NN_weights.parameters(), lr=model_config.meta_opt_lr)       # meta_opt is optimizer for parameters of NN_weights

    #seed for this
    SubsetOperator = SubsetOperator(model_config.batch_size_query, model_config.temp_k_subset, False)

    #seed for this
    SubsetOperatortest = SubsetOperator(model_config.batch_size_query, model_config.temp_k_subset, True)


    # to_device
    ENN = basenet_with_learnable_epinet_and_ensemble_prior(input_feature_size, enn_config.basenet_hidden_sizes, model_config.n_classes, enn_config.exposed_layers, enn_config.z_dim, enn_config.learnable_epinet_hiddens, enn_config.hidden_sizes_prior, enn_config.seed_base, enn_config.seed_learnable_epinet, enn_config.seed_prior_epinet, enn_config.alpha)


    loss_fn_init = nn.CrossEntropyLoss()
    optimizer_init = optim.Adam(ENN.parameters(), lr=model_config.init_train_lr, weight_decay=model_config.init_train_weight_decay)
    # ------- seed for this training
    # ------- train ENN on initial training data  # save the state - ENN_initial_state  # define a separate optimizer for this # how to sample z's ---- separately for each batch
    # ------- they also sampled the data each time and not a dataloader - kind of a bootstrap

    for i in range(model_config.n_train_init):
        ENN.train()
        for (inputs, labels) in dataloader_train:
            z = torch.randn(8)   #set seed for this  #set to_device for this
            optimizer_init.zero_grad()
            outputs = ENN(inputs,z)
            loss = loss_fn_init(outputs, labels)
            loss.backward()
            optimizer_init.step()



    # Predictor =       # model for which we will evaluate recall   # load pretrained weights for the Predictor or train it




    for epoch in range(model_config.n_epoch):
        train(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device, NN_weights, meta_opt, SubsetOperator, ENN, Predictor)

    test(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device,  NN_weights, SubsetOperatortest, ENN, Predictor)

def train(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device, NN_weights, meta_opt, SubsetOperator, ENN, Predictor):
  ENN.train()

  for i in range(train_config.n_train_iter):    # Should we do this multiple times or not
    start_time = time.time()
    x_pool, y_pool = next(iter(dataloader_pool))
    pool_weights = NN_weights(x_pool)   #pool_weights has shape [pool_size,1]
    pool_weights_t = pool_weights.t()  #convert pool_weights to shape [1, pool_size]

    #set seed
    soft_k_vector = SubsetOperator(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]
    soft_k_vector_squeeze = soft_k_vector.squeeze()


    z_pool = torch.randn(8) # set seed for z #set to device
    x_pool_label_ENN_logits = ENN(x_pool,z_pool)  #use here complete dataset
    x_pool_label_ENN_probabilities = F.softmax(logits, dim=1) #see if dim=1 is correct
    x_pool_label_ENN_categorical = distributions.Categorical(x_pool_label_ENN_probabilities)
    x_pool_label_ENN = x_pool_label_ENN_categorical.sample() # set seed for labels           # Do we need to take aeverages over multiple z's here? - No, do this for multiple z's



    ENN_opt = torch.optim.Adam(ENN.parameters(), lr=train_config.ENN_opt_lr)

                                                                              #copy_initial_weights - will be important if we are doing multisteps  # how to give initial training weights to ENN -- this is resolved , if we use same instance of the model everywhere - weights get stored
    meta_opt.zero_grad()
    with higher.innerloop_ctx(ENN, ENN_opt, copy_initial_weights=False) as (fnet, diffopt):

      for _ in range(train_config.n_ENN_iter):

        for (idx_batch, x_batch, label_batch) in dataloader_pool_train:

          z_pool_train = torch.randn(8)

          fnet_logits = fnet(x_batch, z_pool_train)    # Forward pass (outputs are logits) #DEFINE fnet sampler through fnet
          batch_log_probs = F.log_softmax(fnet_logits, dim=1)     #see if here dim=1 is correct or not   # Apply log-softmax to get log probabilities


          batch_weights = soft_k_vector_squeeze[idx_batch]        # Retrieve weights for the current batch
          x_batch_label_ENN = x_pool_label_ENN[idx_batch]         # Retrieve labels for the current batch

          # Calculate loss
          ENN_loss = weighted_nll_loss(batch_log_probs,x_batch_label_ENN,batch_weights)       #expects log_probabilities as inputs    #CHECK WORKING OF THIS

          diffopt.step(ENN_loss)

      #derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters
      meta_loss = var_recall_estimator(fnet, dataloader_test, Predictor)      #see where does this calculation for meta_loss happens that is it outside the innerloop_ctx or within it
      meta_loss.backward()

    meta_opt.step()
    # log all important metrics and also save model configs

def test(train_config, dataloader_pool, dataloader_pool_train, dataloader_test, device,  NN_weights, SubsetOperatortest, ENN, Predictor):

  ENN.train()
  x_pool,y_pool = = next(iter(dataloader_pool))                     #corect this with arguments if we needed
  pool_weights = NN_weights(x_pool)   #pool_weights has shape [pool_size,1]
  pool_weights_t = pool_weights.t()  #convert pool_weights to shape [1, pool_size]

  #set seed
  hard_k_vector = SubsetOperatortest(pool_weights_t)     #soft_k_vector has shape  [1,pool_size]
  hard_k_vector_squeeze = soft_k_vector.squeeze()


  ENN_opt = torch.optim.Adam(ENN.parameters(), lr=train_config.ENN_opt_lr)


  with higher.innerloop_ctx(ENN, ENN_opt, track_higher_grads=False) as (fnet, diffopt):

    for _ in range(train_config.n_ENN_iter):

      for (idx_batch, x_batch, label_batch) in dataloader_pool_train:

          z_pool_train = torch.randn(8)

          fnet_logits = fnet(x_batch, z_pool_train)    # Forward pass (outputs are logits) #DEFINE fnet sampler through fnet
          batch_log_probs = F.log_softmax(fnet_logits, dim=1)     #see if here dim=1 is correct or not   # Apply log-softmax to get log probabilities


          batch_weights = hard_k_vector_squeeze[idx_batch]        # Retrieve weights for the current batch
          y_batch = y_pool[idx_batch]         # Retrieve labels for the current batch

          # Calculate loss
          ENN_loss = weighted_nll_loss(batch_log_probs,y_batch,batch_weights)       #expects log_probabilities as inputs    #CHECK WORKING OF THIS

          diffopt.step(ENN_loss)

    meta_loss = var_recall_estimator(fnet, dataloader_test, Predictor, para = {'tau': 0.4}).detach()
    #see what does detach() do and if needed here


  #log and print important things here

# Example usage


dataset_cfg = DatasetConfig("train.csv", "test.csv", "pool.csv", "y_col")
model_cfg = ModelConfig(batch_size_train = 64, batch_size_test = 64, batch_size_query = 100, temp_k_subset = 0.1, hidden_sizes_weight_NN = [50,50], meta_opt_lr = 0.001, n_classes = 2, n_epoch = 10, init_train_lr = 0.001, init_train_weight_decay = 0.1, n_train_init = 20)
train_cfg = TrainConfig(n_train_iter = 15, n_ENN_iter = 15, ENN_opt_lr = 0.001)
enn_cfg = ENNConfig(input_size = 2, basenet_hidden_sizes = [50,50], n_classes = 2, exposed_layers = [False, True], z_dim = 8, learnable_epinet_hiddens = [15,15], hidden_sizes_prior = [5,5], seed_base = 2, seed_learnable_epinet = 1, seed_prior_epinet = 0, alpha = 0.1)

#Predictor = .......

experiment(dataset_cfg, model_cfg, train_cfg, enn_cfg, Predictor)