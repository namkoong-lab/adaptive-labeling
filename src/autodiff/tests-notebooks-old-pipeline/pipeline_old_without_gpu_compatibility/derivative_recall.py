#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  
import higher


# In[ ]:


res = 0
n = 5
h = torch.tensor([0.15 for i in range(n)])
c = torch.tensor([1, 0, 1, 0, 1]) #fix classifier


tau = 0.1
gamma = 0.5
epsilon = 0.7

N_iter = 10 # #z sampled for enn

def sigmoid_gamma(x): #sigmoid func
    return 1/(1+ torch.exp(- gamma*x))
    
def approx_ber(h): #h is n-dim; output is an approx Bernoulli vector with mean h
    n = len(h)
    u = np.array([[np.random.uniform() for i in range(n)] for i in range(2)])
    G = torch.tensor(np.array([-np.log(-np.log(_)) for _ in u]))

    x1 = torch.exp((torch.log(h) + G[0])/tau)
    x2 = torch.exp((torch.log(torch.add(1,-h)) + G[1])/tau)
    x_sum = torch.add(x1,x2)
    x = torch.div(x1,x_sum)
    
    return x


def func(x):
    return approx_ber(x**2)


def Model_pred(X, model): #return output of model 
    prediction = model(X)
    # Predicted class value using argmax
    predicted_class = np.argmax(prediction)
    return predicted_class


def Recall(h, predicted_class): #input is Bernoulli(h) and classifier c, output is recall
    Y_vec = approx_ber(h)
    n = len(h)
    
    #vec1 = sigmoid_gamma(torch.add(approx_ber(h), -epsilon))
    #vec2 = sigmoid_gamma(c)
    
    x = torch.sum(torch.mul(Y_vec, predicted_class))
    y = torch.sum(Y_vec)
    
    return x/y

def var_recall_estimator(fnet, dataloader_test, Predictor):
    predicted_class = Model_pred(dataloader_test, Predictor) #generate labels
    res = []
    res_square = []
    for i in range(N_iter):
        z = torch.randn(i) # sample z
        res.append(Recall(fnet(dataloader_test,z), predicted_class))
        res_square.append(Recall(fnet(dataloader_test,z), predicted_class) ** 2)

    res = torch.tensor(res)
    res_square = torch.tensor(res_square)

    var = torch.mean(res_square) - (torch.mean(res)) ** 2

    return var



##var_recall_estimator(fnet, dataloader_test, Predictor)
#derivative of fnet_parmaeters w.r.t NN (sampling policy) parameters is known - now we need derivative of var recall w.r.t fnet_parameters



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
    z = #sample z from P_z
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

