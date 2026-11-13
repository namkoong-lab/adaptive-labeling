#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  
import higher


# In[ ]:

    
def approx_ber(h, tau): #h is n-dim; output is an approx Bernoulli vector with mean h
    n = len(h)
    u = torch.rand((2, n))
    G = -torch.log(-torch.log(u)) #convert it into Gumbel

    x1 = torch.exp((torch.log(h) + G[0])/tau)
    x2 = torch.exp((torch.log(torch.add(1,-h)) + G[1])/tau)
    x_sum = torch.add(x1,x2)
    x = torch.div(x1,x_sum)
    
    return x



def Model_pred(X_loader, model): #return output of model 
    prediction_list = torch.empty((0, 1), dtype=torch.float32)
    for (x_batch, label_batch) in X_loader:
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)
 
    
    predicted_class = torch.argmax(prediction_list)
    predicted_class = prediction_list >= 0.5 #may need to use the previous code if model predicts probs of two classes
    return predicted_class


def Recall(h, predicted_class, tau): #input is Bernoulli(h) and classifier c, output is recall
    Y_vec = approx_ber(h, tau) #generate random label
    n = len(h)
    
    Y_vec = torch.unsqueeze(Y_vec, 1)

    x = torch.sum(torch.mul(Y_vec, predicted_class))
    y = torch.sum(Y_vec)
    
    return x/y

def var_recall_estimator(fnet, dataloader_test, Predictor, para):
    tau = para['tau']
    N_iter = para['n_iter_var_recall']

    predicted_class = Model_pred(dataloader_test, Predictor) #generate y_pred

    res  = torch.empty((0), dtype=torch.float32)
    res_square  = torch.empty((0), dtype=torch.float32)

    
    for i in range(N_iter):
        z = torch.randn(i) # sample z
        ENN_output_list = torch.empty((0), dtype=torch.float32)
        for (x_batch, label_batch) in dataloader_test:
             

            fnet_logits = fnet(x_batch, z) 
            #fnet_logits_softmax = torch.nn.Softmax(fnet_logits, dim = 1)
            fnet_logits_probs = torch.nn.functional.softmax(fnet_logits, dim=1)
            ENN_output_list = torch.cat((ENN_output_list,fnet_logits_probs[:,1]),0) 
        recall_est = Recall(ENN_output_list, predicted_class, tau)
         
        res = torch.cat((res,(recall_est).view(1)),0)
        res_square = torch.cat((res_square,(recall_est ** 2).view(1)),0)

    var = torch.mean(res_square) - (torch.mean(res)) ** 2
     
    return var



##ignore the below
##var_recall_estimator(fnet, dataloader_test, Predictor)
#derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters


'''
def func(x):
    return approx_ber(x**2)

def sigmoid_gamma(x): #sigmoid func
    return 1/(1+ torch.exp(- gamma*x))

# d_g_d_h is d_recall/d_eta 
# d_h_d_eta from epinet gradient 
def d_g_d_h(h,d_h_d_eta):    
    y = Recall(h) #approx_ber(x) #[0]#
    y.backward()
    return d_h_d_eta * h.grad
    

def h(eta,z): #epinet structure; eta is epinet weight 
    return epinet(eta,z)


g_h_array = []
d_g_d_eta_array = []

for i in range(n_sim):
    z = torch.randn(i)#sample z from P_z
    g_h_array.append(Recall( h(eta,z) ))
    tmp_derivative = d_g_d_h(h(eta,z),d_h_d_eta) #d_h_d_eta comes from epi_net
    d_g_d_eta_array.append(tmp_derivative)
    
g_h_array_mean = np.mean(g_h_array)

#2 \E_{\blue{z}\sim Z} \Bigg(\Big[ g(h(\eta,\blue{z})) - \E_{\red{z} \sim Z} g(h(\eta,\red{z})) \Big] \frac{\partial g(h(\eta,\blue{z}))}{\partial \eta} \Bigg)
d_var_d_eta = 1/n_sim * np.dot([_ - g_h_array_mean for _ in g_h_array], d_g_d_eta_array) #


# In[ ]:


res = 0
n = 8
 
for j in range(100):
    x = torch.tensor([0.15 for i in range(n)], requires_grad = True)
    c = torch.bernoulli(torch.add(x,0.4))
    
    y = Recall(x,c) #approx_ber(x) #[0]#
    y.backward()
    print(x.grad)


# In[ ]:


#### GP task

def Error(h, covariance): #input is mean and covariance matrix
     
    n = len(h)
    
    noise = np.random.normal(size = (1,n))
    A = np.linalg.cholesky(covariance) 
    Z = A.dot(noise) 

    tmp = torch.tensor([(h[i] + Z[i] - c[i])**2 for i in range(n)]) 
    
    return torch.sum(tmp)/n
'''
