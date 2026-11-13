import torch
import gpytorch
import torch.nn as nn

from sample_normal import sample_multivariate_normal

def var_ate_estimator(model, test_x, Predictor, device, n_samples):    #expects test_x = [N,D] and model to be a gpytorch model

    latent_posterior = model(test_x)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([n_samples]))
    prediction = Predictor(test_x).squeeze()
    ate_each_point = torch.subtract(latent_posterior_sample, prediction)
    ate_each_f = torch.mean(ate_each_point, dim=1)
    ate_variance = torch.var(ate_each_f)
    #print("ate_variance:",ate_variance)

    ate_mean = torch.mean(ate_each_f)
    #L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    #print("L_2_loss_mean:", L_2_loss_mean)

    return ate_mean, ate_variance    #shape = scalar, scalar

def ate(test_x, test_y, Predictor, device):    #expects test_x = [N,D] and test_y =[N]
    prediction = Predictor(test_x).squeeze()
    diff = torch.subtract(test_y, prediction)
    return torch.mean(diff)     #shape = scalar

def var_ate_custom_gp_estimator(mu, cov, noise, test_x, Predictor, device, n_samples):      #expects test_x = [N,D], noise=float, mu =[N], cov=[N,N]

    latent_posterior_sample = sample_multivariate_normal(mu, cov, n_samples)       #[n_samples, N]
    prediction = Predictor(test_x).squeeze() #[N]
    ate_each_point = torch.subtract(latent_posterior_sample, prediction) #[n_samples, N]  
    ate_each_f = torch.mean(ate_each_point, dim=1)   #[n_samples]
    ate_variance = torch.var(ate_each_f)       # scalar
    #print("ate_variance:",ate_variance)
    ate_mean = torch.mean(ate_each_f)
    #ate_mean = torch.mean(ate_each_f)+noise    #scalar
    #print("ate_mean:", ate_mean)

    return ate_mean, ate_variance                        # shape = scalar, scalar