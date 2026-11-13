import torch
import gpytorch
import torch.nn as nn




def l2_loss(test_x, test_y, Predictor, device):
    prediction = Predictor(test_x).squeeze()
    #print("prediction:", prediction)
    #print("test_y:", test_y)
    diff_square = torch.square(torch.subtract(test_y, prediction))
    #print("diff_square:", diff_square)
    return torch.mean(diff_square)

def var_l2_loss_estimator(fnet, test_x, Predictor, device, z_dim, stdev_noise):    #expects test_x = [N,D] and model to be a gpytorch model

    res  = torch.empty((0), dtype=torch.float32, device=device)


    for z_pool in range(z_dim): #i is iter in ENN z
        
        #z_pool = torch.randn(z_dim, device=device)     # sample z
        fnet_y = fnet(test_x, z_pool)  #enn output
        prediction = Predictor(test_x)  #prediction output which is mean
        l2_loss_list = torch.square(torch.subtract(fnet_y, prediction)) #l
        l2_est = torch.mean(l2_loss_list) #avg l2 loss for one fixed z
        #print("l2_est:", l2_est)

        res = torch.cat((res,l2_est.view(1)),0) #append E[l2 loss] for a fixed z
        #print("res:",res)

    L_2_loss_variance = torch.var(res)
    L_2_loss_mean = torch.mean(res)+stdev_noise**2



    return L_2_loss_mean, L_2_loss_variance