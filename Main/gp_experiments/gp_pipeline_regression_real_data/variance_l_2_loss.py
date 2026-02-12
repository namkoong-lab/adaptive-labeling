import torch
import gpytorch
import torch.nn as nn

from sample_normal import sample_multivariate_normal


def var_l2_loss_estimator_pg(model, test_x, Predictor, device, n_samples):    #expects test_x = [N,D] and model to be a gpytorch model   
    model_dumi = model.to("cpu")
    test_x_dumi = test_x.to("cpu")
    latent_posterior = model_dumi(test_x_dumi)
    #print("latent_posterior:",latent_posterior)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([n_samples]))
    #print("latent_posterior_sample:",latent_posterior_sample)

    prediction = Predictor(test_x.float()).squeeze()
    #print("prediction:",prediction)
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample.float(), prediction))
    #print("L_2_loss_each_point:",L_2_loss_each_point)

    
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    #print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    #print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_mean, L_2_loss_variance    #shape = scalar, scalar



def var_l2_loss_estimator(model, test_x, Predictor, device, n_samples):    #expects test_x = [N,D] and model to be a gpytorch model   
    latent_posterior = model(test_x)
    print("latent_posterior:",latent_posterior)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([n_samples]))
    #print("latent_posterior_sample:",latent_posterior_sample)

    prediction = Predictor(test_x.float()).squeeze()
    print("prediction:",prediction)
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample.float(), prediction))
    print("L_2_loss_each_point:",L_2_loss_each_point)

    
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    #print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    #print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_mean, L_2_loss_variance    #shape = scalar, scalar

def l2_loss(test_x, test_y, Predictor, device):    #expects test_x = [N,D] and test_y =[N]
    prediction = Predictor(test_x.float()).squeeze()
    diff_square = torch.square(torch.subtract(test_y.float(), prediction))
    return torch.mean(diff_square)     #shape = scalar

def var_l2_loss_custom_gp_estimator(mu, cov, noise, test_x, Predictor, device, n_samples):      #expects test_x = [N,D], noise=float, mu =[N], cov=[N,N]

    latent_posterior_sample = sample_multivariate_normal(mu, cov, n_samples)       #[n_samples, N]
    prediction = Predictor(test_x.float()).squeeze() #[N]
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample.float(), prediction)) #[n_samples, N]  
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)   #[n_samples]
    L_2_loss_variance = torch.var(L_2_loss_each_f)       # scalar
    #print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+noise    #scalar
    #print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_mean, L_2_loss_variance                        # shape = scalar, scalar