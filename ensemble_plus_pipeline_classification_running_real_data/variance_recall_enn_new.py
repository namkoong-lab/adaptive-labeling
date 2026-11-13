import torch
import numpy as np  
import higher








def approx_ber(logits, tau, device): #h is n-dim; output is an approx Bernoulli vector with mean h
    gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))
    gumbels = gumbel_dist.sample(logits.size()).to(logits.device)                   ### Can use torch.clamp(x, min=1, max=3) here - torch.clamp is autodiffable - but we will not face the inf/nan issue as torch.softmax handles it by subtacting maximum value from all the values.
    y_soft = torch.softmax((logits + gumbels) / tau, dim=1)
    y = y_soft[:,1]
    return y




def Model_pred(X_loader, model, device):
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in X_loader:
        prediction = model(x_batch)    # has dimension [batch_size,dim_output] where dim_output might be 1 or 2 for binary classification
        prediction_list = torch.cat((prediction_list,prediction),0)

    if prediction_list.size(1) > 1:
        predicted_class = torch.argmax(prediction_list, dim=1)       #may need to use the previous code if model predicts probs of two classes
    elif prediction_list.size(1) == 1:
        predicted_class = prediction_list >= 0.5
    return predicted_class





def Recall(ENN_logits, predicted_class, tau, device):

    Y_vec = approx_ber(ENN_logits, tau, device)

    Y_vec = torch.unsqueeze(Y_vec, 1)

    x = torch.sum(torch.mul(Y_vec, predicted_class))
    y = torch.sum(Y_vec)
    y = y.clamp(min=1.0)
    return x/y






def Recall_True(dataloader_test, model, device):
    label_list  = torch.empty((0), dtype=torch.float32, device=device)
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)

    for (x_batch, label_batch) in dataloader_test:
        label_list = torch.cat((label_list,label_batch),0)
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)
        #print("x_batch", x_batch)

    if prediction_list.size(1) > 1:
        predicted_class = torch.argmax(prediction_list, dim=1)       #may need to use the previous code if model predicts probs of two classes
    elif prediction_list.size(1) == 1:
        predicted_class = prediction_list >= 0.5

    predicted_class = torch.squeeze(predicted_class)
    print("predicted class:", predicted_class)
    label_list = torch.squeeze(label_list)

    x = torch.sum(torch.mul(label_list, predicted_class))
    y = torch.sum(label_list)

    return x/y


def var_recall_estimator(ENN_base, ENN_prior, dataloader_test, Predictor, device, tau, z_dim, n_samples, n_iter_noise, alpha):

    predicted_class= Model_pred(dataloader_test, Predictor, device)    
    #expected to be a tensor of dim [N,1]
    #dataloader test must have shuffle false
    #torch.manual_seed(seed_var)
    res  = torch.empty((0), dtype=torch.float32, device=device)
    #res_square  = torch.empty((0), dtype=torch.float32, device=device)


    for z_pool in range(z_dim):
        #z_pool = torch.randn(z_dim, device=device)
        ENN_logits = torch.empty((0,2), dtype=torch.float32, device=device)
        for (x_batch, label_batch) in dataloader_test:
            fnet_logits = ENN_base(x_batch, z_pool)+ alpha* ENN_prior(x_batch, z_pool)
            #fnet_logits_probs = torch.nn.functional.softmax(fnet_logits, dim=1) ---- no need of this as logits can work themselves
            ENN_logits = torch.cat((ENN_logits,fnet_logits),dim=0)
        #recall est over multiple Gumbel RV
        recall_est_list = torch.empty((0), dtype=torch.float32, device=device)
        for j in range(n_iter_noise):
            recall_est = Recall(ENN_logits, predicted_class, tau, device).view(1) #use diff seeds for Gumbel
            recall_est_list = torch.cat((recall_est_list, recall_est),0)
        #print("recall_est:", recall_est)
        res = torch.cat((res,torch.mean(recall_est_list).view(1)),0) #append mean of recall over multiple Gumbel
        #print("res:",res)
        #res_square = torch.cat((res_square,(recall_est ** 2).view(1)),0)

    mean_recall = torch.mean(res)
    var_recall = torch.var(res)
 
    return mean_recall, var_recall


