import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import gpytorch
#### Advanced version for training as well


class RBFKernelAdvanced(nn.Module):
    def __init__(self, length_scale_init=0.6931471824645996, variance_init=0.6931471824645996):
        super(RBFKernelAdvanced, self).__init__()
        self.raw_length_scale = nn.Parameter(torch.tensor([length_scale_init], dtype=torch.float))
        self.raw_variance = nn.Parameter(torch.tensor([variance_init], dtype=torch.float))

        self.softplus = nn.Softplus()

    def forward(self, x1, x2):
        #length_scale = self.softplus(self.raw_length_scale)
        #variance = self.softplus(self.raw_variance)
        #length_scale = self.raw_length_scale
        #variance = self.raw_variance
        #sqdist = torch.cdist(x1, x2) ** 2
        length_scale = torch.where(self.raw_length_scale < 0, F.softplus(self.raw_length_scale), self.raw_length_scale)
        variance = torch.where(self.raw_variance < 0, F.softplus(self.raw_variance), self.raw_variance)
        dist_matrix = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0)**2
        return variance * torch.exp(-0.5  * dist_matrix / length_scale ** 2)


class GaussianProcessCholeskyAdvanced(nn.Module):
    def __init__(self, length_scale_init=0.6931471824645996, variance_init=0.6931471824645996, noise_var_init=0.1):
        super(GaussianProcessCholeskyAdvanced, self).__init__()
        self.rbf_kernel = RBFKernelAdvanced(length_scale_init=length_scale_init, variance_init=variance_init)
        self.raw_noise_var = nn.Parameter(torch.tensor([noise_var_init], dtype=torch.float))

        self.softplus = nn.Softplus()

    def forward(self, x_train, y_train, w_train, x_test, stabilizing_constant=1e-5, noise= None):

        # Apply weights only to non-diagonal elements
        # Not using above noise, but instead as initiated and tuned

        noise_var = torch.where(self.raw_noise_var < 0, F.softplus(self.raw_noise_var), self.raw_noise_var)
        #noise_var = self.softplus(self.raw_noise_var)
        #noise_var = self.raw_noise_var
        
        K = self.rbf_kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + stabilizing_constant * torch.eye(x_train.size(0), device=x_train.device)
        #K = self.kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + 1e-5 * torch.eye(x_train.size(0), device=x_train.device)
        non_diag_mask = 1 - torch.eye(K.size(-2), K.size(-1), device=x_train.device)
        weight_matrix = w_train.unsqueeze(-1) * w_train.unsqueeze(-2)
        weighted_K =  K * (non_diag_mask * weight_matrix + (1 - non_diag_mask))



        K_s = self.rbf_kernel(x_train, x_test)
        weighted_K_s = torch.diag(w_train)@K_s

        K_ss = self.rbf_kernel(x_test, x_test) + stabilizing_constant * torch.eye(x_test.size(0), device=x_test.device)

        L = torch.linalg.cholesky(weighted_K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        mu = weighted_K_s.t().matmul(alpha).squeeze(-1)

        v = torch.linalg.solve(L, weighted_K_s)
        cov = K_ss - v.t().matmul(v)

        return mu, cov

    def nll(self, x_train, y_train, stabilizing_constant=1e-5):

        noise_var = torch.where(self.raw_noise_var < 0, F.softplus(self.raw_noise_var), self.raw_noise_var)

        K = self.rbf_kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + stabilizing_constant * torch.eye(x_train.size(0), device=x_train.device)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        nll = 0.5 * y_train.dot(alpha.flatten())
        nll += torch.log(torch.diag(L)).sum()
        nll += 0.5 * len(x_train) * torch.log(torch.tensor(2 * torch.pi, device=nll.device))
        #print("nll:", nll)
        return nll