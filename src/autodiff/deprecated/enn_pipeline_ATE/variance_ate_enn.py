import torch
import gpytorch
import torch.nn as nn




def ate(test_x, test_y, Predictor, device):
    prediction = Predictor(test_x).squeeze()
    #print("prediction:", prediction)
    #print("test_y:", test_y)
    ate = torch.subtract(test_y, prediction)
    return torch.mean(ate)

def var_ate_estimator(fnet, test_x, Predictor, device, z_dim, n_samples, stdev_noise):    #expects test_x = [N,D] and model to be a gpytorch model

    res  = torch.empty((0), dtype=torch.float32, device=device)


    for i in range(n_samples): #i is iter in ENN z
        
        z_pool = torch.randn(z_dim, device=device)     # sample z
        fnet_y = fnet(test_x, z_pool)  #enn output
        prediction = Predictor(test_x)  #prediction output which is mean
        ate_list = torch.subtract(fnet_y, prediction) #l
        ate_est = torch.mean(ate_list) #avg ate for one fixed z

        res = torch.cat((res,ate_est.view(1)),0) #append E[l2 loss] for a fixed z
        #print("res:",res)

    ate_variance = torch.var(res)
    ate_mean = torch.mean(res)



    return ate_mean, ate_variance