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


######################################## Introducing new objective





# def approx_ber(logits, tau, device): #h is n-dim; output is an approx Bernoulli vector with mean h
#     gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))
#     gumbels = gumbel_dist.sample(logits.size()).to(logits.device)                   ### Can use torch.clamp(x, min=1, max=3) here - torch.clamp is autodiffable - but we will not face the inf/nan issue as torch.softmax handles it by subtacting maximum value from all the values.
#     y_soft = torch.softmax((logits + gumbels) / tau, dim=1)
#     y = y_soft[:,1]
#     return y




def Model_pred(X_loader, model, device):
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in X_loader:
        prediction = model(x_batch)   # has dimension [batch_size,dim_output] where dim_output is assumed to be 1 and is the probability of y=1
        prediction_list = torch.cat((prediction_list,prediction),0)

    print("prediction_list_2:", prediction_list)
    return prediction_list





# def Recall(ENN_logits, predicted_class, tau, device):

#     Y_vec = approx_ber(ENN_logits, tau, device)

#     Y_vec = torch.unsqueeze(Y_vec, 1)

#     x = torch.sum(torch.mul(Y_vec, predicted_class))
#     y = torch.sum(Y_vec)
#     y = y.clamp(min=1.0)
#     return x/y






def Recall_True(dataloader_test, model, device):
    label_list  = torch.empty((0), dtype=torch.float32, device=device)
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)


    # for (x_batch, label_batch) in dataloader_test:
    #     label_list = torch.cat((label_list,label_batch),0)
    #     prediction = model(x_batch)
    #     prediction_list = torch.cat((prediction_list,prediction),0)
    #     #print("x_batch", x_batch)

    # if prediction_list.size(1) > 1:
    #     predicted_class = torch.argmax(prediction_list, dim=1)       #may need to use the previous code if model predicts probs of two classes
    # elif prediction_list.size(1) == 1:
    #     predicted_class = prediction_list >= 0.5


    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in dataloader_test:
        label_list = torch.cat((label_list,label_batch),0)
        prediction = model(x_batch)    # has dimension [batch_size,dim_output] where dim_output is assumed to be 1 and is the probability of y=1
        prediction_list = torch.cat((prediction_list,prediction),0)    
    label_list = label_list.unsqueeze(1)
    #print("prediction_list",prediction_list)
    #print("label_list", label_list)
    
    #predictor_loss_list =  (1-label_list)*torch.log(prediction_list) + label_list*torch.log(1-prediction_list)
    predictor_loss_list =  (label_list)*torch.log(prediction_list) + (1-label_list)*torch.log(1-prediction_list)
    #predictor_loss = -torch.mean(predictor_loss_list)
    predictor_loss = -torch.mean(predictor_loss_list)
    #print("predictor_loss:",predictor_loss)
    #print("predictor_loss_2:",predictor_loss_2)


    return predictor_loss


def var_recall_estimator(ENN_base, ENN_prior, dataloader_test, Predictor, device, tau, z_dim, n_samples, n_iter_noise, alpha):

    predicted_probabilities = Model_pred(dataloader_test, Predictor, device)
    #print("predicted probabilties var", predicted_probabilities)    
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
        ENN_logit_probs = torch.nn.functional.softmax(ENN_logits, dim=1)
        #print("ENN_logit_probs", ENN_logit_probs)
        #print("ENN_logit_probs[:,1:]", ENN_logit_probs[:,1:])
        #print("ENN_logit_probs[:,0:1]", ENN_logit_probs[:,0:1])


        model_pred_loss_list =  ENN_logit_probs[:,1:] * torch.log(predicted_probabilities) +  ENN_logit_probs[:,0:1]* torch.log(1-predicted_probabilities)
        model_pred_loss = -torch.mean(model_pred_loss_list)
        res = torch.cat((res,model_pred_loss.view(1)),0)
        
    #print("res:", res)
    mean_recall = torch.mean(res)
    var_recall = torch.var(res)
 
    return mean_recall, var_recall

