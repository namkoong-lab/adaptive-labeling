#version: https://github.com/namkoong-lab/adaptive_sampling/blob/7160a71c0d30b12cb5824899d301ce1a79b5cc1b/src/autodiff/gp_pipeline_regression/polyadic_sampler.py

import torch
from torch import Tensor
import gpytorch
import math
import matplotlib.pyplot as plt
import os
import wandb

# no_training_points, input_dim, ENN or GP or TNP, no_anchor_points(geq 3), distance_factor -> will also determine the length parameter of GPs - but we should also try other parameters

import argparse
import typing
from dataclasses import dataclass
from scipy.stats import percentileofscore


def get_binary_from_y(y_list):
    '''
    input is a list of data, we combine them, then compute the percentile of data, then sample probability from it
    '''
    combined_list = torch.cat(y_list)
    binary_data_list = [torch.bernoulli(torch.tensor([percentileofscore(combined_list, i, kind='strict')/100 for i in y])) for y in y_list]
    return binary_data_list

@dataclass
class PolyadicSamplerConfig:
    def __init__(self, no_train_points: int, no_test_points: int, no_pool_points: int, model_name: str, no_anchor_points: int, input_dim: int, stdev_scale: float, stdev_pool_scale: float, scaling_factor = None, scale_by_input_dim = None, model = None, stdev_blr_w = None, stdev_blr_noise = None, logits =  None, if_logits = None, if_logits_only_pool = None, plot_folder = None, no_train_clusters = 1):
        self.no_train_points = no_train_points
        self.no_test_points = no_test_points
        self.no_pool_points = no_pool_points
        self.model_name = model_name
        self.no_anchor_points = no_anchor_points
        self.input_dim = input_dim
        self.stdev_scale = stdev_scale
        self.stdev_pool_scale = stdev_pool_scale
        self.scaling_factor = scaling_factor
        self.scale_by_input_dim = scale_by_input_dim
        self.model = model
        self.stdev_blr_w = stdev_blr_w
        self.stdev_blr_noise = stdev_blr_noise
        self.logits = logits
        self.if_logits = if_logits
        self.if_logits_only_pool = if_logits_only_pool
        self.plot_folder = plot_folder
        self.no_train_clusters = no_train_clusters


def x_sampler(no_train_points, no_test_points, no_pool_points, no_anchor_points=3, input_dim = 1, stdev_scale = 0.2, stdev_pool_scale = 0.5, scaling_factor=None, scale_by_input_dim = True, logits = None, if_logits= False, if_logits_only_pool= False, no_train_clusters = 1):

  # no_anchor_points \geq 3
  # stdev_scale -  controls the spread around each anchor point
  # stdev_pool_scale  -  stdev of pool scaled by this amount
  # scale_by_input_dim - just to control the oveerall spread of points within (-2,2)
  # if_logits - if True supply the "logits" - to determine the probability of each anchor point around which the pint is drawn - otherwise drawn uniformly
  # logits - will determine the probability of each anchor point around which the pint is drawn


  if scaling_factor != None:
    scaling_factor =  scaling_factor
  elif scale_by_input_dim:
    scaling_factor = no_anchor_points*math.sqrt(input_dim)   # Ideally use this to ensure we are within training range
  else:
    scaling_factor = no_anchor_points  # This might go beyond training range

  # distance between two anchor points = sqrt(n)/scaling_factor = [1/no_anchor_pts] or [sqrt(n)/no_anchor_pts] (depending on if scale_by_input_dim =True or False respectively)
  # max spread around anchor pts = 2*sqrt(n)*stdev_train_test_scale/scaling_factor = [2*stdev_train_test_scale/(no_anchor_points)] or [2*sqrt(n)*stdev_train_test_scale/(no_anchor_points)]
  # min distance between spread of adjacent anchor points  =  [1/no_anchor_pts *(1-4*stdev_train_test_scale) ] or [sqrt(n)/no_anchor_pts *(1-4*stdev_train_test_scale) ]
  # ideal GP params for which this dataset will be interesting (or hard to solve is)  - length scale = now the GP s should remain relevant around anchor point but not between two anchor points - length_scale ~ [stdev_train_test_scale/(no_anchor_points)] or [sqrt(n)*stdev_train_test_scale/(no_anchor_points)]
  # then scale*e^{-1/2} will be var of close points and scale*e^{-1/(2*stdev_scale^2)} ~ 0 across adjacent anchor points
  # length_scale ~ [stdev_train_test_scale/(no_anchor_points)] or [sqrt(n)*stdev_train_test_scale/(no_anchor_points)]

  anchor_x = torch.randn(no_anchor_points, input_dim)
  scaling_anchors = math.sqrt(input_dim) * torch.arange(no_anchor_points)
  norms = torch.sqrt(torch.sum(anchor_x**2, dim=1, keepdim=True))
  anchor_x = (anchor_x / norms) * scaling_anchors.unsqueeze(1)


  train_sample_idx = torch.randint(0, no_train_clusters, (no_train_points,))        # or you can change it to take training points from more anchor points - currently only one anchor point

  if if_logits:
    assert logits != None, "Input the logits - as if_logits is True"
    assert logits.size(0) == no_anchor_points, "no_anchor_points does not match the logits dimension"
    print("Anchor points are sampled using logits")
    probabilities = torch.softmax(logits, dim=0)
    test_sample_idx = torch.multinomial(probabilities, no_test_points, replacement=True)
    pool_sample_idx = torch.multinomial(probabilities, no_pool_points, replacement=True)
  
  elif if_logits_only_pool:
    assert logits != None, "Input the logits - as if_logits is True"
    assert logits.size(0) == no_anchor_points, "no_anchor_points does not match the logits dimension"
    print("Anchor points for pool are sampled using logits")
    probabilities = torch.softmax(logits, dim=0)
    test_sample_idx = torch.randint(0, no_anchor_points, (no_test_points,))
    pool_sample_idx = torch.multinomial(probabilities, no_pool_points, replacement=True)


  else:
    print("Anchor points are sampled uniformly")
    test_sample_idx = torch.randint(0, no_anchor_points, (no_test_points,))
    pool_sample_idx = torch.randint(0, no_anchor_points, (no_pool_points,))

  train_x = (anchor_x[train_sample_idx]+torch.randn(no_train_points, input_dim)*stdev_scale)/scaling_factor
  test_x = (anchor_x[test_sample_idx]+torch.randn(no_test_points, input_dim)*stdev_scale)/scaling_factor
  pool_x = (anchor_x[pool_sample_idx]+torch.randn(no_pool_points, input_dim)*stdev_scale*stdev_pool_scale)/scaling_factor


  # Returning test_sample_idx, pool_sample_idx so that we can now recover the ideal solutions - ex. one from each cluster
  return train_x, test_x, pool_x, test_sample_idx, pool_sample_idx


class CustomizableGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, base_kernel, likelihood):
        super(CustomizableGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = base_kernel
        self.likelihood = likelihood

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def y_sampler(model_name, train_x, test_x, pool_x, stdev_scale, no_anchor_points, model=None, stdev_blr_w= None, stdev_blr_noise= None):
  if model_name=="GP":
    if model==None:
      print("Using default GP model")
      mean_constant = 0.0  # Mean of the GP

      # this length_parameter is set according to the x generated - to make this setting more interesting and hard - We should also try random length_scales

      # Also don't set stdev_train_test_scale>1 and also dont set it too low - some interesting values will be 0.1, 0.5, 1.0, 1.5
      length_scale = stdev_scale/(no_anchor_points)    # if scale_by_input_dim = True, otherwise  length_scale =  sqrt(input_dim)*stdev_train_test_scale/(no_anchor_points)     #length_scale = [[0.5,0.5]]   # Length scale of the RBF kernel if you want different length parameter for each dimension
      output_scale = 0.69
      noise_std = 0.1      # Standard deviation of the noise


      mean_module = gpytorch.means.ConstantMean()
      base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())    #base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))      # use this if you want to set the different length parameter for each dimension of the kernel
      likelihood = gpytorch.likelihoods.GaussianLikelihood()


      mean_module.constant = mean_constant
      base_kernel.base_kernel.lengthscale = length_scale
      base_kernel.outputscale = output_scale
      likelihood.noise_covar.noise = noise_std**2


      x=torch.cat([train_x, test_x, pool_x], dim =0)


      y = torch.zeros(x.size(0))

      model = CustomizableGPModel(x, y, mean_module, base_kernel, likelihood)

      model.eval()
      likelihood.eval()
      with torch.no_grad():
        prior_dist = likelihood(model(x))
        y_new = prior_dist.sample()


      train_y = y_new[:train_x.size(0)]
      test_y = y_new[train_x.size(0):train_x.size(0)+test_x.size(0)]
      pool_y = y_new[train_x.size(0)+test_x.size(0):]

    else:
      print("Using given GP model")

      x=torch.cat([train_x, test_x, pool_x], dim =0)

      y = torch.zeros(x.size(0))

      model.eval()
      model.likelihood.eval()

      model.set_train_data(inputs = x, targets = y , strict=False)



      with torch.no_grad():
        prior_dist = model.likelihood(model(x))
        y_new = prior_dist.sample()


      train_y = y_new[:train_x.size(0)]
      test_y = y_new[train_x.size(0):train_x.size(0)+test_x.size(0)]
      pool_y = y_new[train_x.size(0)+test_x.size(0):]


  elif model_name=="blr":
    assert stdev_blr_w != None, "Specify the stdev_blr_w"
    assert stdev_blr_noise != None,  "Specify the stdev_blr_noise"
    print("Using blr model")
    w = torch.randn(train_x.size(1))*stdev_blr_w
    train_y = torch.matmul(train_x,w)+ torch.randn(train_x.size(0))*stdev_blr_noise
    test_y = torch.matmul(test_x,w)+ torch.randn(test_x.size(0))*stdev_blr_noise
    pool_y = torch.matmul(pool_x,w)+ torch.randn(pool_x.size(0))*stdev_blr_noise

    
  binary_data = get_binary_from_y([train_y,test_y, pool_y])

  return train_y, test_y, pool_y, binary_data[0], binary_data[1], binary_data[2]

# define combined everything



def generate_dataset(
    no_train_points, no_test_points, no_pool_points,
    model_name, # gp or blr model, trained TNP model
    no_anchor_points=3, input_dim = 1,
    stdev_scale = 0.2, stdev_pool_scale = 0.5, scaling_factor=None, scale_by_input_dim = True, # controlling the spread of x points
    model=None, # gpytorch gp model to sample y given x
    stdev_blr_w = None, stdev_blr_noise = None, # controlling blr sampling y given x
    logits = None, if_logits= False, if_logits_only_pool= False, # logits control how many points each anchor point is centered around
    no_train_clusters = 1 #controls how many clusters are there in the training points
    ):

    train_x, test_x, pool_x, test_sample_idx, pool_sample_idx = x_sampler(no_train_points, no_test_points, no_pool_points, no_anchor_points, input_dim, stdev_scale, stdev_pool_scale, scaling_factor, scale_by_input_dim, logits, if_logits, if_logits_only_pool, no_train_clusters)
    train_y, test_y, pool_y, train_y_binary, test_y_binary, pool_y_binary = y_sampler(model_name, train_x, test_x, pool_x, stdev_scale, no_anchor_points, model, stdev_blr_w, stdev_blr_noise)

    return train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx, train_y_binary, test_y_binary, pool_y_binary


def set_data_parameters_and_generate(polyadic_sampler_config):

    train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx, train_y_binary, test_y_binary, pool_y_binary = generate_dataset(
       polyadic_sampler_config.no_train_points, polyadic_sampler_config.no_test_points, polyadic_sampler_config.no_pool_points,
       polyadic_sampler_config.model_name,
       polyadic_sampler_config.no_anchor_points, polyadic_sampler_config.input_dim, polyadic_sampler_config.stdev_scale,
       polyadic_sampler_config.stdev_pool_scale, polyadic_sampler_config.scaling_factor, polyadic_sampler_config.scale_by_input_dim, polyadic_sampler_config.model,
       polyadic_sampler_config.stdev_blr_w, polyadic_sampler_config.stdev_blr_noise, polyadic_sampler_config.logits, polyadic_sampler_config.if_logits, polyadic_sampler_config.if_logits_only_pool,
       polyadic_sampler_config.no_train_clusters
    )



    if train_x.size(1) == 1:    
      fig1 = plt.figure()
      plt.scatter(train_x,  train_y, label='Train')
      plt.scatter(test_x,  test_y, label='Test')

      # Annotate each point in pool_x with its index
      for i, (x, y) in enumerate(zip(pool_x, pool_y)):
          plt.annotate(i, (x, y))

      plt.scatter(pool_x, pool_y, label='Pool')
      plt.legend()
      wandb.log({"env_plot_with_pool_indexes": wandb.Image(fig1)})
      plt.show()
      plt.close(fig1)

      
      fig2 = plt.figure()
      plt.scatter(train_x,  train_y, label='Train')
      plt.scatter(train_x,  train_y_binary, label='Train_binary')
      plt.legend()
      wandb.log({"Train plot": wandb.Image(fig2)})
      plt.show()
      plt.close(fig2)

      fig3 = plt.figure()
      plt.scatter(test_x,  test_y, label='Test')
      plt.scatter(test_x,  test_y_binary, label='Test_binary')
      plt.legend()
      wandb.log({"Test plot": wandb.Image(fig3)})
      plt.show()
      plt.close(fig3)
        
      fig4 = plt.figure()
      plt.scatter(pool_x,  pool_y, label='Pool')
      plt.scatter(pool_x,  pool_y_binary, label='Pool_binary')
      plt.legend()
      wandb.log({"Pool plot": wandb.Image(fig4)})
      plt.show()
      plt.close(fig4)



    return train_x, train_y, test_x, test_y, pool_x, pool_y, test_sample_idx, pool_sample_idx, train_y_binary, test_y_binary, pool_y_binary





