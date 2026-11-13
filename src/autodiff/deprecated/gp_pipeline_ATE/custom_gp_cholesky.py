import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import gpytorch

class GaussianProcessCholesky(nn.Module):
    def __init__(self, kernel):
        super(GaussianProcessCholesky, self).__init__()
        self.kernel = kernel

    def forward(self, x_train, y_train, w_train, x_test, stabilizing_constant=1e-5, noise_var=1e-4):

        # Apply weights only to non-diagonal elements

        K = self.kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + stabilizing_constant * torch.eye(x_train.size(0), device=x_train.device)
        non_diag_mask = 1 - torch.eye(K.size(-2), K.size(-1), device=x_train.device)
        weight_matrix = w_train.unsqueeze(-1) * w_train.unsqueeze(-2)
        weighted_K =  K * (non_diag_mask * weight_matrix + (1 - non_diag_mask))



        K_s = self.kernel(x_train, x_test)
        weighted_K_s = torch.diag(w_train)@K_s

        K_ss = self.kernel(x_test, x_test) + stabilizing_constant * torch.eye(x_test.size(0), device=x_train.device)

        L = torch.linalg.cholesky(weighted_K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        mu = weighted_K_s.t().matmul(alpha).squeeze(-1)

        v = torch.linalg.solve(L, weighted_K_s)
        cov = K_ss - v.t().matmul(v)

        return mu, cov

class RBFKernel(nn.Module):
    def __init__(self, length_scale= 0.6931471824645996, output_scale = 0.6931471824645996):
        super(RBFKernel, self).__init__()
        self.length_scale = length_scale
        self.output_scale = output_scale
    def forward(self, x1, x2):
        dist_matrix = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0)**2
        return self.output_scale*torch.exp(-0.5 * dist_matrix / self.length_scale**2)