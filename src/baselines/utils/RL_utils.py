#!/user/bw2762/.conda/envs/testbed_2/bin/python

from typing import Callable, NamedTuple
import sys
import numpy as np
import pandas as pd
import plotnine as gg

import time

# from acme.utils.loggers.terminal import TerminalLogger
import dataclasses
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import warnings
warnings.filterwarnings('ignore')

#@title Neural Testbed imports
import neural_testbed
from neural_testbed.agents import factories as agent_factories
from neural_testbed.agents.factories.sweeps import real_data_2 as agent_sweeps
from neural_testbed import base
from neural_testbed import generative
from neural_testbed import leaderboard
from neural_testbed import UQ_data
from acme.utils.loggers.csv import CSVLogger
from neural_testbed import agents as enn_agents

from enn import networks
from enn import datasets
from enn import supervised
from enn import utils



def logits_to_label_BinaryClass(x:chex.Array):
    """
    get labels from binary logits
    """
    labels = jnp.array([[jnp.argmax(jax.nn.softmax(logits))] for logits in x])
    return labels

def recall_BinaryClass(predicted_labels:chex.Array
                       ,true_labels:chex.Array):
    """
    calculate the reall with given predicted and true labels 
    """
    TP = jnp.sum((predicted_labels == 1) & (true_labels == 1))
    FN = jnp.sum((predicted_labels == 0) & (true_labels == 1))
    # print(labels==y)
    # numerator = jnp.sum(labels==y)
    # denominator = jnp.sum(y)
    return TP / (TP + FN)




def sample_recall_BinaryClass(Dataset: base.Data,
                                enn_sampler: base.EpistemicSampler,
                                key: chex.PRNGKey):
    """
    MC estimate of the recall mean and Var with given features, true labels, and a enn sampler
    """
    start = time.time()
    M = 100
    print("sample recall variance")
    print("M = "+str(M))
    print("calib set size = "+str(len(Dataset.x)))
    keys = jax.random.split(key,M)
    recall_list = jnp.array([recall_BinaryClass(logits_to_label_BinaryClass(enn_sampler(Dataset.x,keys[i])),Dataset.y) for i in range(M)])
    # print(recall_list)
    recall_mean = jnp.mean(recall_list)
    recall_var = jnp.var(recall_list)

    end = time.time()
    print("total time ="+str(end-start))
    return recall_mean,recall_var

def approximated_recall_estimate_BinaryClass(Dataset: base.Data,
                                enn_sampler: base.EpistemicSampler,
                                key: chex.PRNGKey,
                                seed: int):
    """
    a recall estimator using the Gumble softmax parameterization 
    """
    tau = 0.1
    epsilon = 0.1
    gamma = 1

    # key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key,3)
    x = Dataset.x   #features
    y = Dataset.y   #true labels
    y_pred = enn_sampler(x,subkeys[0])  #pred labels logits

    n = len(x)

    g1 = jax.random.gumbel(subkeys[1],shape=n)
    g2 = jax.random.gumbel(subkeys[2],shape=n)

    temp1 = jnp.exp((jnp.log(y_pred)+g1)/tau)
    temp2 = jnp.exp((jnp.log(1-y_pred)+g2)/tau)
    Y = temp1/(temp1+temp2)

    sigmoid = lambda x,gamma:  1/(1+jnp.exp(-gamma*x))

    sigma_y = sigmoid(Y - epsilon)
    sigma_x = sigmoid(y)
    recall = jnp.sum(sigma_x*sigma_y)/jnp.sum(sigma_y)

    return recall


def extract_posterior(experiment):
    """
    extract the flattened posterior parameters (learnable network) of a given epinet
    """
    init_training_state = experiment.state
    init_params = init_training_state.params
    # init_network_state = init_training_state.network_state

    params_list = []
    params_list.append(init_params['train_epinet/~/mlp/~/linear_0']['w'])
    params_list.append(init_params['train_epinet/~/mlp/~/linear_0']['b'])
    params_list.append(init_params['train_epinet/~/mlp/~/linear_1']['w'])
    params_list.append(init_params['train_epinet/~/mlp/~/linear_1']['b'])
    params_list.append(init_params['train_epinet/~/mlp/~/linear_2']['w'])
    params_list.append(init_params['train_epinet/~/mlp/~/linear_2']['b'])
    flattened_params = jax.numpy.concatenate([param.flatten() for param in params_list])
    # num_params_posterior = len(flattened_params)
    
    return flattened_params


def split_dataset(key, dataset, train_frac=0.6, calib_frac=0.2):
    """
    Splits the dataset into training, calibration, and test sets.

    Args:
    data (np.ndarray): The dataset to split.
    labels (np.ndarray): The labels for the dataset.
    train_frac (float): The fraction of the dataset to use for training.
    calib_frac (float): The fraction of the dataset to use for calibration.

    Returns:
    tuple: Tuple containing training, calibration, and test sets.
    """
    # Shuffle the data and labels in unison
    num_data = len(dataset.x)
    p = jnp.arange(num_data)
    p = jax.random.permutation(key,p)
    data, labels = dataset.x[p], dataset.y[p]

    # Calculate split indices
    train_end = int(train_frac * num_data)
    calib_end = train_end + int(calib_frac * num_data)

    # Split the data and labels
    train_data, train_labels = data[:train_end], labels[:train_end]
    calib_data, calib_labels = data[train_end:calib_end], labels[train_end:calib_end]
    first_batch_data, first_batch_labels = data[calib_end:], labels[calib_end:]

    # Convert to JAX arrays if necessary
    train = datasets.ArrayBatch(x=train_data, y=train_labels)
    calib = datasets.ArrayBatch(x=calib_data, y=calib_labels)
    first_batch = datasets.ArrayBatch(x=first_batch_data, y=first_batch_labels)

    return train, calib, first_batch