#!/user/bw2762/.conda/envs/testbed_2/bin/python

import numpy as np

import tensorflow as tf

import haiku as hk

import jax
import jax.numpy as jnp

import neural_testbed
from neural_testbed.agents import factories as agent_factories
from neural_testbed.agents.factories.sweeps import real_data_2 as agent_sweeps
# from neural_testbed.agents.factories import epinet
from neural_testbed_test_1.neural_testbed.RL_stuff.factories_epinet_v2 import make_agent_v2, EpinetConfig_v2
from neural_testbed_test_1.neural_testbed.RL_stuff.enn_agents_v2 import extract_enn_sampler_v2
from neural_testbed_test_1.neural_testbed.UQ_data.data_modules_2 import generate_problem_v2
from RL_utils import extract_posterior, sample_recall_BinaryClass
from neural_testbed import base
from neural_testbed import generative
from neural_testbed import leaderboard
from neural_testbed import UQ_data
from acme.utils.loggers.csv import CSVLogger
from neural_testbed import agents as enn_agents

from neural_testbed_test_1.neural_testbed.RL_stuff.subset_sampling import sample_subset_continuous,sample_subset,top_k,weighted_reservoir_sampling_get_key

from enn import networks
from enn import datasets
from enn import supervised
from enn import utils

import torch

import gymnasium as gym
from gymnasium import spaces


import stable_baselines3
# from stable_baselines3 import PPO

EPSILON = jnp.finfo(jnp.float32).tiny

######load dataset 

dataset_name = 'eicu'
path_train = '/user/bw2762/UQ_implementation_shared/datasets/eicu_train_final.csv'
path_test = '/user/bw2762/UQ_implementation_shared/datasets/eicu_test_final.csv'
label_name = 'EVENT_LABEL'
num_classes = 2
tau =10
seed = 1
temperature = 0.01
noise_std = 1.
sampler_type = 'global'

problem = generate_problem_v2(path_train,path_test,label_name,dataset_name,sampler_type,num_classes,tau,seed,temperature,noise_std)

train_data = problem.train_data

# test_data = problem.test_data

print(len(train_data.x))

n_samples = len(train_data.x)

print(train_data)

subset_ind = jax.random.choice(jax.random.PRNGKey(0),n_samples,(3000,),replace = False)

subset_data = datasets.ArrayBatch(x=train_data.x[subset_ind], y=train_data.y[subset_ind])
print(len(subset_data))

print(subset_data)


# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface."""

#     metadata = {"render_modes": ["human"], "render_fps": 30}

#     def __init__(self, arg1, arg2):
#         super().__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Discrete(3)
#         # Example for using image as input (channel-first; channel-last also works):
#         self.observation_space = spaces.Box(low=0, high=255,
#                                             shape=(3, 2, 2), dtype=np.uint8)

#     def step(self, action):
#         ...
#         return observation, reward, terminated, truncated, info

#     def reset(self, seed=None, options=None):
#         ...
#         return observation, info

#     def render(self):
#         ...

#     def close(self):
#         ...

class UQ_epinet_ENV(gym.Env):

    T = 5
    # num_params_posterior = 500
    
    def __init__(self,
               first_batch,
               dataset,
               calibration_dataset,
               problem,
               agent_config,
               batch_num,
               seed = 0,
            #    tempreture = 0.1
               ):
        super().__init__()
        self.batch_num = batch_num
        self.first_batch = first_batch
        self.problem = problem
        self.dataset = dataset
        self.calibration_dataset = calibration_dataset
        self.agent_config = agent_config
        self.seed = seed
        # self.tempreture = tempreture

        self.rng = hk.PRNGSequence(seed)
        self.key = jax.random.PRNGKey(seed)
        self.t = 0

        num_dataset = len(dataset.x)
        ###

        self.agent = make_agent_v2(agent_config)
        self.experiment = self.agent(self.first_batch,self.problem.prior_knowledge)
    

        enn_sampler = extract_enn_sampler_v2(self.experiment)
        recall_mean , recall_var = sample_recall_BinaryClass(self.calibration_dataset, enn_sampler,self.key)

        self.recall_var = recall_var

        flattened_posterior = extract_posterior(self.experiment)
        self.num_params_posterior = len(flattened_posterior)

        self.action_space = spaces.Box(low=EPSILON, high=jnp.inf, shape=(num_dataset,))
        self.observation_space = spaces.Box(low=EPSILON, high=jnp.inf, shape=(self.num_params_posterior,))
        print("INITIALIZED")

    def _get_obs(self):
        posterior = extract_posterior(self.experiment)
        return posterior
    
    def _get_info(self):
        return None


    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        print("START RESET")
        super().reset(seed=seed, options=options)

        if seed is None:
            seed = self.seed

        self.rng = hk.PRNGSequence(seed)
        self.t = 0
        self.agent = make_agent_v2(self.agent_config)
        self.experiment = self.agent(self.first_batch,self.problem.prior_knowledge) # discuss the seed
        
        enn_sampler = extract_enn_sampler_v2(self.experiment)
        recall_mean , recall_var = sample_recall_BinaryClass(self.calibration_dataset, enn_sampler,next(self.rng))

        self.recall_var = recall_var

        obs = self._get_obs()
        print("FINISH RESET")
        return obs
        ###TODO

    def step(self, action):
        # key = jax.random.PRNGKey(self.seed)
        # key = next(self.key)
        # key = self.key
        # keys = jax.random.split(key,3)

        # batch = top_k(self.dataset,action,self.batch_num) #WRS
        print("START STEP")
        r = weighted_reservoir_sampling_get_key(action,next(self.rng)) #WRS
        print(len(r))
        batch = top_k(self.dataset,r,self.batch_num)
        print(len(batch.x))
        print(len(batch.y))
        training_state = self.experiment.state
        rng = next(self.experiment.rng)
        training_state, loss_metrics = self.experiment._sgd_step(self.experiment.pure_trainers[0].pure_loss,
                            training_state,
                            batch,
                            rng)
        self.experiment.state = training_state
        observation = self._get_obs()

        enn_sampler = extract_enn_sampler_v2(self.experiment)
        recall_mean , recall_var = sample_recall_BinaryClass(self.calibration_dataset, enn_sampler,next(self.rng))
        reward = -(recall_var - self.recall_var)
        self.recall_var = recall_var

        self.t += 1
        if self.t >= self.T:
            terminated = True
        
        truncated = False
        info = self._get_info()
        print("FINISH STEP")

        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass

    def close(self):
        pass

