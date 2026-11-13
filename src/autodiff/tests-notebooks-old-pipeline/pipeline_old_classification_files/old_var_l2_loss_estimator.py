#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  


def l2_loss(dataloader_test, Predictor, device):
    res  = torch.empty((0), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in dataloader_test:
        prediction = Predictor(x_batch) 
        diff_square = torch.square(torch.subtract(label_batch, prediction))
        res = torch.cat((res,diff_square),0) 
    return torch.mean(res)

 

def var_l2_loss_estimator(fnet, dataloader_test, Predictor, device, para):
    z_dim = para['z_dim']
    N_iter =  para['N_iter']
    if_print =  para['if_print']
    seed = para['seed_var_l2']
    torch.manual_seed(seed)

    res  = torch.empty((0), dtype=torch.float32, device=device)
    res_square  = torch.empty((0), dtype=torch.float32, device=device)

    
    for i in range(N_iter): #i is iter in ENN z
        #each noise compute 1 l2 loss then average 
        z_pool = torch.randn(z_dim, device=device)# sample z

        list_l2_loss_fixed_z = torch.empty((0), dtype=torch.float32, device=device) #an array with l2 loss (over diff noise) for fixed z

        
        for j in range(para['N_iter_noise']):
            l2_loss_list = torch.empty((0), dtype=torch.float32, device=device) #a list with fixed noise, results with total l2 loss
            for (x_batch, label_batch) in dataloader_test:
                fnet_logits = fnet(x_batch, z_pool)  #enn output
                prediction = Predictor(x_batch)  #prediction output which is mean 

                noise_label = para['sigma_noise'] * torch.randn(prediction.shape) #generate Gaussian noise added to the label
                fnet_logits = fnet_logits + noise_label #add noise to fnet_logits
                l2_list_temp = torch.square(torch.subtract(fnet_logits, prediction)) #l2 los

                l2_loss_list = torch.cat((l2_loss_list,l2_list_temp),0) 
            #print("i:",i)
            l2_est = torch.mean(l2_loss_list) #avg l2 loss for one fixed noise vector
            #print("recall_est:", recall_est)
            l2_est = (l2_est).view(1)
            list_l2_loss_fixed_z = torch.cat((list_l2_loss_fixed_z,l2_est),0)  #append results with diff noise vector

         
        l2_est_fixed_z = torch.mean(list_l2_loss_fixed_z) #avg over noise; obtain E[l2 loss] for a fixed z
        res = torch.cat((res,l2_est_fixed_z.view(1)),0) #append E[l2 loss] for a fixed z
        #print("res:",res)
        res_square = torch.cat((res_square,(l2_est_fixed_z ** 2).view(1)),0) # (E[l2 loss])^2 for a fixed z
        
    #print("res_square:", res_square)
    var = torch.mean(res_square) - (torch.mean(res)) ** 2
    if if_print == 1:
        print('l2 list', res)
        print("var of l2_loss:",var)
        print("mean of l2_loss",  torch.mean(res))
    return var