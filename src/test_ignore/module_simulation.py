#!/user/ct3064/.conda/envs/uq-env/bin/python

# -*- coding: utf-8 -*-
"""trying_directly_UQ.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FXSZhHcCdnScP0ykM-DPZp2J3Lfwx2kz
"""
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/ct3064/.conda/envs/uq-env/lib

import os
#@title General imports
from typing import Callable, NamedTuple
import sys
import numpy as np
import pandas as pd
import plotnine as gg

# from acme.utils.loggers.terminal import TerminalLogger
import dataclasses
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
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

#from neural_testbed_test_1.neural_testbed.UQ_data.data_modules_2 import generate_problem_v2
from data_modules_2 import generate_problem_v2
#@title Valid problem_id problems and implemented agents:

#problem_id problems
print('All possible values for problem_id:', UQ_data.UQ)

# Implemented agents:
print('All implemeted agents:', agent_sweeps.get_implemented_agents())

kw_dict = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        sep = arg.find('=')
        key, value = arg[:sep], arg[sep + 1:]
        kw_dict[key] = value

print(kw_dict)

agent_name = kw_dict['agent_name']
k_val = float(kw_dict['k_val'])

import csv

def append_to_csv(data, file_path):
    """Appends a row of data to the specified CSV file."""
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


directory = './module_evaluations/'
# Define the CSV file path.

path = './datasets/'
benchmark_name = 'FDB' # start with ACS first
dataset_name = 'fraudecom'
only_important_features = False # only important features for selection biased training sets

if only_important_features:
    train_features = 'important_features'
    feature_tag = "_only_important_feature"
else:
    train_features = "all_features"
    feature_tag = ""

label_name = 'EVENT_LABEL'
num_classes = 2
# tau: indicator of marginal or joint log loss (1: marginal log loss, 10: joint log loss)
tau =10
temperature = 0.01
noise_std = 1.
#sampler_type = 'global'


# FIX: currently using both cd and ood combined as test data
#path_test = f"{path}{benchmark_name}_train_test/{dataset_name}_test_final.csv"
# to be filled
generic_test_path = f"{path}biased_latest_5/{benchmark_name}_test/{dataset_name}/{train_features}"
incomplete_test_file = f"{dataset_name}_random_prop_score_"
cd_path_test = f"{generic_test_path}/{incomplete_test_file}selected" # selected
ood_path_test = f"{generic_test_path}/{incomplete_test_file}not_selected" # not selected
test_paths = { "cd": cd_path_test, "ood": ood_path_test}
log_loss_vals = {"marginal": 1, "joint": 10}


directory = './module_evaluations/'+dataset_name+'/'+agent_name
if not os.path.exists(directory):
    os.makedirs(directory)

# key signals ood or cd, def_testpath is the corresponding default test file path

seed = 2 # seed of selection bias dataset generation
first_agent_config = True
# defining train and test paths 
# training on selection biased data
path_train = f"./datasets/biased_latest_5/{benchmark_name}_train/{dataset_name}/{train_features}/{dataset_name}_random_prop_score_not_selected2_{str(k_val)}{feature_tag}_simple.csv"
ood_test_path = f"{test_paths['ood']}{str(seed)}_{str(k_val)}_simple.csv"
cd_test_path = f"{test_paths['cd']}{str(seed)}_{str(k_val)}_simple.csv"

csv_file_path = directory+'/'+'population_'+str(k_val)+"_outputs.csv"
csv_file_path_2 = directory +'/'+'population_'+str(k_val)+"_runlog.csv"
csv_file_path_3 = directory +'/'+'population_'+str(k_val)+"_agent_config.csv"

# append_to_csv(["data_name","tau","sweep_count"],csv_file_path_2)
# append_to_csv(["dataset_name","agent_name", "seed", "tau", "kl_estimate_dyadic", "kl_estimate_dyadic_stdev","kl_estimate_uniform","kl_estimate_uniform_stdev", "train_acc_1","train_ece_1","test_acc_1", "test_ece_1", "train_acc_2","train_ece_2", "test_acc_2",  "test_ece_2","agent_config"], csv_file_path)
append_to_csv(["data_name","sweep_count"],csv_file_path_2)


arr_1 = ['dataset_name', 'Running_agent', 'seed']
arr_1.extend(['kl_estimate_dyadic_tau_1_ood', 'kl_estimate_uniform_tau_1_ood', 'kl_estimate_dyadic_tau_10_ood', 'kl_estimate_uniform_tau_10_ood', 'kl_estimate_dyadic_tau_1_cd', 'kl_estimate_uniform_tau_1_cd',  'kl_estimate_dyadic_tau_10_cd', 'kl_estimate_uniform_tau_10_cd'])
arr_1.extend(['test_acc_dyadic_tau_1_ood', 'test_acc_uniform_tau_1_ood', 'test_acc_dyadic_tau_10_ood', 'test_acc_uniform_tau_10_ood', 'test_acc_dyadic_tau_1_cd', 'test_acc_uniform_tau_1_cd',  'test_acc_dyadic_tau_10_cd', 'test_acc_uniform_tau_10_cd'])
arr_1.extend(['test_ece_dyadic_tau_1_ood', 'test_ece_uniform_tau_1_ood', 'test_ece_dyadic_tau_10_ood', 'test_ece_uniform_tau_10_ood', 'test_ece_dyadic_tau_1_cd', 'test_ece_uniform_tau_1_cd',  'test_ece_dyadic_tau_10_cd', 'test_ece_uniform_tau_10_cd'])
arr_1.extend(['train_acc_dyadic_tau_1_ood', 'train_acc_uniform_tau_1_ood', 'train_acc_dyadic_tau_10_ood', 'train_acc_uniform_tau_10_ood'])
arr_1.extend(['train_ece_dyadic_tau_1_ood', 'train_ece_uniform_tau_1_ood', 'train_ece_dyadic_tau_10_ood', 'train_ece_uniform_tau_10_ood'])
arr_1.extend(['kl_estimate_dyadic_tau_1_ood_stdev', 'kl_estimate_uniform_tau_1_ood_stdev', 'kl_estimate_dyadic_tau_10_ood_stdev', 'kl_estimate_uniform_tau_10_ood_stdev', 'kl_estimate_dyadic_tau_1_cd_stdev', 'kl_estimate_uniform_tau_1_cd_stdev', 'kl_estimate_dyadic_tau_10_cd_stdev', 'kl_estimate_uniform_tau_10_cd_stdev'])
                    


append_to_csv(arr_1, csv_file_path)

# defining problems
# data train (sample biased) train, test ood, k=1   marginal log loss, dyadic sampling
problem_1 = generate_problem_v2(path_train,ood_test_path,label_name,dataset_name,num_classes,'local',log_loss_vals["marginal"],seed,temperature,noise_std)
# data train (sample biased) train, test ood, k=1   marginal log loss, uniform sapling
problem_2 = generate_problem_v2(path_train,ood_test_path,label_name,dataset_name,num_classes,'global',log_loss_vals["marginal"],seed,temperature,noise_std)
# data train (sample biased) train, test ood, k=10    joint log loss, dyadic sampling
problem_3 = generate_problem_v2(path_train,ood_test_path,label_name,dataset_name,num_classes,'local',log_loss_vals["joint"],seed,temperature,noise_std)
# data train (sample biased) train, test ood, k=10    joint log loss, uniform sampling
problem_4 = generate_problem_v2(path_train,ood_test_path,label_name,dataset_name,num_classes,'global',log_loss_vals["joint"],seed,temperature,noise_std)
# data train (sample biased) train, test cd, k=1    marginal log loss, dyadic sampling
problem_5 = generate_problem_v2(path_train,cd_test_path,label_name,dataset_name,num_classes,'local',log_loss_vals["marginal"],seed,temperature,noise_std)
# data train (sample biased) train, test cd, k=1    marginal log loss, uniform sampling
problem_6 = generate_problem_v2(path_train,cd_test_path,label_name,dataset_name,num_classes,'global',log_loss_vals["marginal"],seed,temperature,noise_std)
# data train (sample biased) train, test cd, k=10    joint log loss, dyadic sampling
problem_7 = generate_problem_v2(path_train,cd_test_path,label_name,dataset_name,num_classes,'local',log_loss_vals["joint"],seed,temperature,noise_std)
# data train (sample biased) train, test cd, k=10    joint log loss, uniform sampling
problem_8 = generate_problem_v2(path_train,cd_test_path,label_name,dataset_name,num_classes,'global',log_loss_vals["joint"],seed,temperature,noise_std)

#for agent_name in [ 'ensemble', 'dropout', 'epinet',  'bbb', 'mlp', 'ensemble+', 'hypermodel']:
paper_agent_1 = agent_sweeps.get_paper_agent(agent_name)
sweep_count = 0

for configuration in paper_agent_1.sweep():

    sweep_count = sweep_count + 1
    agent_1 = paper_agent_1.ctor(configuration)
    print(f'Running agent={agent_name} on problem={dataset_name,k_val}....')

    config_new_1 = agent_1.config
    config_new_1.eval_batch_size =1000
    config_new_1.train_log_freq = 5
    config_new_1.eval_log_freq = 5

    print(configuration)

    curve_directory = 'learning_curves/'
    # warning: tau hardcoded
    if agent_name == 'mlp':
        curve_directory += 'mlp/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.l2_weight_decay)
    elif agent_name == 'dropout':   
        curve_directory += 'dropout/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.length_scale)+ "_" + str(configuration.dropout_rate)
    elif agent_name == 'ensemble':   
        curve_directory += 'ensemble/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.l2_weight_decay)
    elif agent_name == 'ensemble+':   
        curve_directory += 'ensemble+/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.l2_weight_decay)+ "_" + str(configuration.prior_scale)     
    elif agent_name == 'hypermodel':   
        curve_directory += 'hypermodel/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.l2_weight_decay)+ "_" + str(configuration.prior_scale) 
    elif agent_name == 'epinet':   
        curve_directory += 'epinet/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.num_batches) + "_" + str(configuration.l2_weight_decay)+ "_" + str(configuration.prior_scale)
    elif agent_name == 'sgmcmc':   
        curve_directory += 'sgmcmc/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.learning_rate) + "_" + str(configuration.prior_variance)+ "_" + str(configuration.momentum_decay) 
    elif agent_name == 'bbb':   
        curve_directory += 'bbb/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.sigma_1) + "_" + str(configuration.sigma_2)+ "_" + str(configuration.mixture_scale)+ "_" + str(configuration.num_batches)         
    elif agent_name == 'deep_kernel':   
        curve_directory += 'deep_kernel/' + dataset_name+ "_"+agent_name+ "_" + str(tau)+ "_"+ str(configuration.learning_rate) + "_" + str(configuration.scale_factor) + "_" + str(configuration.sigma_squared_factor)
    

    config_new_1.logger = CSVLogger(curve_directory,'',0., False,30)

    agent_new_1 = enn_agents.make_learning_curve_enn_agent(config_new_1, problem_1, 1000, 0)


    enn_sampler_1 = agent_new_1(problem_1.train_data, problem_1.prior_knowledge)
    
    quality_1 = problem_1.evaluate_quality(enn_sampler_1)
    quality_2 = problem_2.evaluate_quality(enn_sampler_1)
    quality_3 = problem_3.evaluate_quality(enn_sampler_1)
    quality_4 = problem_4.evaluate_quality(enn_sampler_1)
    quality_5 = problem_5.evaluate_quality(enn_sampler_1)
    quality_6 = problem_6.evaluate_quality(enn_sampler_1)
    quality_7 = problem_7.evaluate_quality(enn_sampler_1)
    quality_8 = problem_8.evaluate_quality(enn_sampler_1)
    
    print(f'Config of agent={configuration} has Quality_dyadic ={quality_1} and Quality_uniform = {quality_2}....')

    Running_agent = agent_name
    seed = configuration.seed
    
    
    arr = [dataset_name, Running_agent, seed]
    
    
    kl_estimate_dyadic_tau_1_ood = quality_1.kl_estimate
    kl_estimate_uniform_tau_1_ood = quality_2.kl_estimate
    kl_estimate_dyadic_tau_10_ood = quality_3.kl_estimate
    kl_estimate_uniform_tau_10_ood = quality_4.kl_estimate
    kl_estimate_dyadic_tau_1_cd = quality_5.kl_estimate
    kl_estimate_uniform_tau_1_cd = quality_6.kl_estimate
    kl_estimate_dyadic_tau_10_cd = quality_7.kl_estimate
    kl_estimate_uniform_tau_10_cd = quality_8.kl_estimate
    
    arr.extend([kl_estimate_dyadic_tau_1_ood, kl_estimate_uniform_tau_1_ood, kl_estimate_dyadic_tau_10_ood, kl_estimate_uniform_tau_10_ood, kl_estimate_dyadic_tau_1_cd, kl_estimate_uniform_tau_1_cd,  kl_estimate_dyadic_tau_10_cd, kl_estimate_uniform_tau_10_cd])
    
    
    test_acc_dyadic_tau_1_ood = quality_1.extra['test_acc']
    test_acc_uniform_tau_1_ood = quality_2.extra['test_acc']
    test_acc_dyadic_tau_10_ood = quality_3.extra['test_acc']
    test_acc_uniform_tau_10_ood = quality_4.extra['test_acc']
    test_acc_dyadic_tau_1_cd = quality_5.extra['test_acc']
    test_acc_uniform_tau_1_cd = quality_6.extra['test_acc']
    test_acc_dyadic_tau_10_cd = quality_7.extra['test_acc']
    test_acc_uniform_tau_10_cd = quality_8.extra['test_acc']
    
    arr.extend([test_acc_dyadic_tau_1_ood, test_acc_uniform_tau_1_ood, test_acc_dyadic_tau_10_ood, test_acc_uniform_tau_10_ood, test_acc_dyadic_tau_1_cd, test_acc_uniform_tau_1_cd,  test_acc_dyadic_tau_10_cd, test_acc_uniform_tau_10_cd])
    
    
    test_ece_dyadic_tau_1_ood = quality_1.extra['test_ece']
    test_ece_uniform_tau_1_ood = quality_2.extra['test_ece']
    test_ece_dyadic_tau_10_ood = quality_3.extra['test_ece']
    test_ece_uniform_tau_10_ood = quality_4.extra['test_ece']
    test_ece_dyadic_tau_1_cd = quality_5.extra['test_ece']
    test_ece_uniform_tau_1_cd = quality_6.extra['test_ece']
    test_ece_dyadic_tau_10_cd = quality_7.extra['test_ece']
    test_ece_uniform_tau_10_cd = quality_8.extra['test_ece']
    
    arr.extend([test_ece_dyadic_tau_1_ood, test_ece_uniform_tau_1_ood, test_ece_dyadic_tau_10_ood, test_ece_uniform_tau_10_ood, test_ece_dyadic_tau_1_cd, test_ece_uniform_tau_1_cd,  test_ece_dyadic_tau_10_cd, test_ece_uniform_tau_10_cd])
    
    
    
    
    
    train_acc_dyadic_tau_1 = quality_1.extra['train_acc']
    train_acc_uniform_tau_1 = quality_2.extra['train_acc']
    train_acc_dyadic_tau_10 = quality_3.extra['train_acc']
    train_acc_uniform_tau_10 = quality_4.extra['train_acc']
    
    arr.extend([train_acc_dyadic_tau_1, train_acc_uniform_tau_1, train_acc_dyadic_tau_10, train_acc_uniform_tau_10])
    
    
    train_ece_dyadic_tau_1 = quality_1.extra['train_ece']
    train_ece_uniform_tau_1 = quality_2.extra['train_ece']
    train_ece_dyadic_tau_10 = quality_3.extra['train_ece']
    train_ece_uniform_tau_10 = quality_4.extra['train_ece']
    
    arr.extend([train_ece_dyadic_tau_1, train_ece_uniform_tau_1, train_ece_dyadic_tau_10, train_ece_uniform_tau_10])
    

    
    kl_estimate_dyadic_tau_1_ood_stdev = quality_1.extra['kl_estimate_std']
    kl_estimate_uniform_tau_1_ood_stdev = quality_2.extra['kl_estimate_std']
    kl_estimate_dyadic_tau_10_ood_stdev = quality_3.extra['kl_estimate_std']
    kl_estimate_uniform_tau_10_ood_stdev = quality_4.extra['kl_estimate_std']
    kl_estimate_dyadic_tau_1_cd_stdev = quality_5.extra['kl_estimate_std']
    kl_estimate_uniform_tau_1_cd_stdev = quality_6.extra['kl_estimate_std']
    kl_estimate_dyadic_tau_10_cd_stdev = quality_7.extra['kl_estimate_std']
    kl_estimate_uniform_tau_10_cd_stdev = quality_8.extra['kl_estimate_std']
    
    # construct outpust.csv
    arr.extend([kl_estimate_dyadic_tau_1_ood_stdev, kl_estimate_uniform_tau_1_ood_stdev, kl_estimate_dyadic_tau_10_ood_stdev, kl_estimate_uniform_tau_10_ood_stdev, kl_estimate_dyadic_tau_1_cd_stdev, kl_estimate_uniform_tau_1_cd_stdev,  kl_estimate_dyadic_tau_10_cd_stdev, kl_estimate_uniform_tau_10_cd_stdev])
    append_to_csv(arr, csv_file_path)
    
    # construct runlog.csv
    append_to_csv([dataset_name, sweep_count],csv_file_path_2)
    
    
    # construct agent_config.csv
    attr_list = []
    attr_val_list = []
    for attr_name, attr_value in vars(configuration).items():
        attr_list.append(attr_name)
        attr_val_list.append(attr_value)
    
    if first_agent_config:    
        append_to_csv(attr_list,csv_file_path_3)
        first_agent_config = False
    append_to_csv(attr_val_list, csv_file_path_3)
        
        
    
            