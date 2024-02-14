# -*- coding: utf-8 -*-
"""gradient_recall_new_function_and_check_score_gradient.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-NK2a0ctPmp6Uwozxr2317JyqfTC1nDF
"""

import torch
import pandas
import numpy

#### NEW FUNCTION

def approx_ber(logits, tau, device): #h is n-dim; output is an approx Bernoulli vector with mean h
    gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))
    gumbels = gumbel_dist.sample(logits.size()).to(logits.device)                   ### Can use torch.clamp(x, min=1, max=3) here - torch.clamp is autodiffable - but we will not face the inf/nan issue as torch.softmax handles it by subtacting maximum value from all the values.
    y_soft = torch.softmax((logits + gumbels) / tau, dim=1)
    y = y_soft[:,1]
    return y



def Model_pred(X_loader, model, device):
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in X_loader:
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)


    predicted_class = torch.argmax(prediction_list)       ## what is need of this??
    predicted_class = prediction_list >= 0.5
    return predicted_class


def Recall(ENN_logits, predicted_class, tau, device):
    Y_vec = approx_ber(ENN_logits, tau, device)

    Y_vec = torch.unsqueeze(Y_vec, 1)

    x = torch.sum(torch.mul(Y_vec, predicted_class))
    y = torch.sum(Y_vec)
    return x/y

def Recall_True(dataloader_test, model, device):
    label_list  = torch.empty((0), dtype=torch.float32, device=device)
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)

    for (x_batch, label_batch) in dataloader_test:
        label_list = torch.cat((label_list,label_batch),0)
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)

    #predicted_class = torch.argmax(prediction_list)                             ### why is this needed??
    predicted_class = prediction_list >= 0.5
    predicted_class = torch.squeeze(predicted_class, 1)

    x = torch.sum(torch.mul(label_list, predicted_class))
    y = torch.sum(label_list)

    return x/y

def var_recall_estimator(fnet, dataloader_test, Predictor, device, para):
    tau = para['tau']
    z_dim = para['z_dim']
    N_iter =  para['N_iter']
    if_print =  para['if_print']
    predicted_class = Model_pred(dataloader_test, Predictor, device)

    res  = torch.empty((0), dtype=torch.float32, device=device)
    res_square  = torch.empty((0), dtype=torch.float32, device=device)


    for i in range(N_iter):
        z_pool = torch.randn(z_dim, device=device)
        ENN_logits = torch.empty((0,2), dtype=torch.float32, device=device)
        for (x_batch, label_batch) in dataloader_test:
            fnet_logits = fnet(x_batch, z_pool)
            #fnet_logits_probs = torch.nn.functional.softmax(fnet_logits, dim=1) ---- no need of this as logits can work themselves
            ENN_logits = torch.cat((ENN_logits,fnet_logits),dim=0)
        recall_est = Recall(ENN_logits, predicted_class, tau, device)
        res = torch.cat((res,(recall_est).view(1)),0)
        res_square = torch.cat((res_square,(recall_est ** 2).view(1)),0)

    var = torch.mean(res_square) - (torch.mean(res)) ** 2
    if if_print == 1:
        print('recall list', res)
        print("var of recall:",var)
        print("mean of recall",  torch.mean(res))
    return var


# In[ ]:


#res = 0
#n = 5
#h = torch.tensor([0.15 for i in range(n)])
#c = torch.tensor([1, 0, 1, 0, 1]) #fix classifier


#tau = 0.1
#gamma = 0.5
#epsilon = 0.7


##ignore the below
##var_recall_estimator(fnet, dataloader_test, Predictor)
#derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters

import torch
import pandas
import numpy
from torch.nn.functional import cosine_similarity

### Reinforce estimator for the gradient

#derivative of E(recall estimator) w.r.t

N=1000
N_iter = 1000
predicted_class = torch.randint(0, 2, (N,))
#print(predicted_class)
random_logits = torch.randn(N, 2, requires_grad=True)
#print(random_logits)

gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

soft_recall_vector  = torch.empty((0), dtype=torch.float32)
hard_derivative_recall_vector =  torch.empty((0,N), dtype=torch.float32)
tau = 0.1

for i in range(N_iter):
  gumbels = gumbel_dist.sample(random_logits.size())
  #print("gumbels:",gumbels)
  logits_perturbed = random_logits + gumbels
  y_soft = torch.softmax(logits_perturbed / tau, dim=1)
  y_soft_final = y_soft[:,1]
  #print("y_soft_final:", y_soft_final)
  y_hard = torch.argmax(logits_perturbed,dim =1)
  #print("y_hard:",y_hard)

  #y_soft_final = torch.unsqueeze(y_soft_final, 1)
  #y_hard = torch.unsqueeze(y_hard, 1)

  Recall_soft = torch.sum(torch.mul(y_soft_final, predicted_class))/torch.sum(y_soft_final)
  #print("Recall_soft:", Recall_soft)
  soft_recall_vector  = torch.cat((soft_recall_vector,(Recall_soft).view(1)),0)
  #print("soft_recall_vector:", soft_recall_vector)

  Recall_hard_numerator = torch.sum(torch.mul(y_hard, predicted_class))
  #print("Recall_hard_numerator:", Recall_hard_numerator)
  Recall_hard_denominator = torch.sum(y_hard)
  #print("Recall_hard_denominator:",Recall_hard_denominator)
  hard_derivative_recall =  torch.empty((0), dtype=torch.float32)

  for j in range(N):
       hard_derivative_recall_one =  ((Recall_hard_numerator - y_hard[j]*predicted_class[j]+predicted_class[j])/(Recall_hard_denominator-y_hard[j]+1))-((Recall_hard_numerator - y_hard[j]*predicted_class[j])/(Recall_hard_denominator-y_hard[j]))
       #print("hard_derivative_recall_one:", hard_derivative_recall_one)
       hard_derivative_recall  = torch.cat((hard_derivative_recall,(hard_derivative_recall_one).view(1)),0)
       #print("hard_derivative_recall:", hard_derivative_recall)
       hard_derivative_recall_unseq = hard_derivative_recall.unsqueeze(0)
  hard_derivative_recall_vector =   torch.cat((hard_derivative_recall_vector,hard_derivative_recall_unseq),0)
  #print("hard_derivative_recall_vector:", hard_derivative_recall_vector)

soft_racall_final = torch.mean(soft_recall_vector)
soft_racall_final.backward()
soft_racall_final_gradient = random_logits.grad[:,1]

#print(hard_derivative_recall_vector)
hard_recall_gradient_vector = hard_derivative_recall_vector.mean(dim=0)
probabilities = torch.softmax(random_logits, dim=1)
probabilities_multiplied = torch.prod(probabilities, dim=1)
hard_recall_gradient_vector_success = probabilities_multiplied * hard_recall_gradient_vector
print(hard_recall_gradient_vector_success)
print(soft_racall_final_gradient)

cos_sim = cosine_similarity(soft_racall_final_gradient.unsqueeze(0), hard_recall_gradient_vector_success.unsqueeze(0))

cos_sim

####OLD FUNCTION


def approx_ber(h, tau, device): #h is n-dim; output is an approx Bernoulli vector with mean h
    n = len(h)
    u = torch.rand((2, n), device=device)
    G = -torch.log(-torch.log(u))
    x1 = torch.exp((torch.log(h) + G[0])/tau)
    x2 = torch.exp((torch.log(torch.add(1,-h)) + G[1])/tau)
    x_sum = torch.add(x1,x2)
    x = torch.div(x1,x_sum)
    return x

 #m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
 #       g = m.sample()
 #       scores = scores + g
 #scores = scores + torch.log(khot_mask)
 #           onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)

def Model_pred(X_loader, model, device):
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in X_loader:
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)


    predicted_class = torch.argmax(prediction_list)
    predicted_class = prediction_list >= 0.5
    return predicted_class


def Recall(h, predicted_class, tau, device):
    Y_vec = approx_ber(h, tau, device)
    n = len(h)

    Y_vec = torch.unsqueeze(Y_vec, 1)

    x = torch.sum(torch.mul(Y_vec, predicted_class))
    y = torch.sum(Y_vec)
    return x/y

def Recall_True(dataloader_test, model, device):
    label_list  = torch.empty((0), dtype=torch.float32, device=device)
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)

    for (x_batch, label_batch) in dataloader_test:
        label_list = torch.cat((label_list,label_batch),0)
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)

    predicted_class = torch.argmax(prediction_list)
    predicted_class = prediction_list >= 0.5
    predicted_class = torch.squeeze(predicted_class, 1)

    x = torch.sum(torch.mul(label_list, predicted_class))
    y = torch.sum(label_list)

    return x/y

def var_recall_estimator(fnet, dataloader_test, Predictor, device, para):
    tau = para['tau']
    z_dim = para['z_dim']
    N_iter =  para['N_iter']
    if_print =  para['if_print']
    predicted_class = Model_pred(dataloader_test, Predictor, device)

    res  = torch.empty((0), dtype=torch.float32, device=device)
    res_square  = torch.empty((0), dtype=torch.float32, device=device)


    for i in range(N_iter):
        z_pool = torch.randn(z_dim, device=device)
        ENN_output_list = torch.empty((0), dtype=torch.float32, device=device)
        for (x_batch, label_batch) in dataloader_test:
            fnet_logits = fnet(x_batch, z_pool)
            fnet_logits_probs = torch.nn.functional.softmax(fnet_logits, dim=1)
            ENN_output_list = torch.cat((ENN_output_list,fnet_logits_probs[:,1]),0)
        recall_est = Recall(ENN_output_list, predicted_class, tau, device)
        res = torch.cat((res,(recall_est).view(1)),0)
        res_square = torch.cat((res_square,(recall_est ** 2).view(1)),0)

    var = torch.mean(res_square) - (torch.mean(res)) ** 2
    if if_print == 1:
        print('recall list', res)
        print("var of recall:",var)
        print("mean of recall",  torch.mean(res))
    return var


# In[ ]:


#res = 0
#n = 5
#h = torch.tensor([0.15 for i in range(n)])
#c = torch.tensor([1, 0, 1, 0, 1]) #fix classifier


#tau = 0.1
#gamma = 0.5
#epsilon = 0.7


##ignore the below
##var_recall_estimator(fnet, dataloader_test, Predictor)
#derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters

