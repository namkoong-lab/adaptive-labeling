#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np  
import higher




#N_iter = 100 # #z sampled for enn


    
def approx_ber(h, tau, seed, device): #h is n-dim; output is an approx Bernoulli vector with mean h
    torch.manual_seed(seed)
    n = len(h)
    #u = np.array([[np.random.uniform() for i in range(n)] for i in range(2)])      # can replace by  u = torch.rand((2, n), device=device)
    #G = torch.tensor(np.array([-np.log(-np.log(_)) for _ in u])).to(device)
    u = torch.rand((2, n), device=device)
    G = -torch.log(-torch.log(u))
    #print("G:",G[0:2,0:2])
    #print("G.shap:",G.shape)
    #ua = torch.rand((2, n), device=device)
    #Ga = -torch.log(-torch.log(ua))
    #print("Ga:",Ga[0:2,0:2])
    #print("Ga.shap:",Ga.shape)
    #print("G[0]:",G[0])
    #print("h:",h)
    x1 = torch.exp((torch.log(h) + G[0])/tau)
    #print("x1:",x1)
    x2 = torch.exp((torch.log(torch.add(1,-h)) + G[1])/tau)
    #print("x2:",x2)
    x_sum = torch.add(x1,x2)
    #print("x_sum:",x_sum)
    x = torch.div(x1,x_sum)
    #print("x:",x)
    return x



def Model_pred(X_loader, model, device): #return output of model 
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)
    for (x_batch, label_batch) in X_loader:
        #x_batch = x_batch.to(device)
        #label_batch = label_batch.to(device)
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)
 
    
    predicted_class = torch.argmax(prediction_list)
    predicted_class = prediction_list >= 0.5 #may need to use the previous code if model predicts probs of two classes
    return predicted_class


def Recall(h, predicted_class, tau, seed, device): #input is Bernoulli(h) and classifier c, output is recall
    torch.manual_seed(seed)
    Y_vec = approx_ber(h, tau, seed, device) #generate random label
    n = len(h)
    
    Y_vec = torch.unsqueeze(Y_vec, 1)

    x = torch.sum(torch.mul(Y_vec, predicted_class))
    #print("x:",x)
    y = torch.sum(Y_vec)
    #print("y:",y)
    return x/y

def Recall_True(dataloader_test, model, device): #input is dataloader_test and classifier/ model c, output is true recall given labels
    label_list  = torch.empty((0), dtype=torch.float32, device=device)
    prediction_list = torch.empty((0, 1), dtype=torch.float32, device=device)

    for (x_batch, label_batch) in dataloader_test:
        label_list = torch.cat((label_list,label_batch),0)
        prediction = model(x_batch)
        prediction_list = torch.cat((prediction_list,prediction),0)

    predicted_class = torch.argmax(prediction_list)
    predicted_class = prediction_list >= 0.5 #may need to use the previous code if model predicts probs of two classes
    predicted_class = torch.squeeze(predicted_class, 1)

    x = torch.sum(torch.mul(label_list, predicted_class))
    y = torch.sum(label_list)

    return x/y

def var_recall_estimator(fnet, dataloader_test, Predictor, device, para):
    tau = para['tau']
    z_dim = para['z_dim']
    N_iter =  para['N_iter']
    if_print =  para['if_print']
    seed = para['seed_var_recall']
    N_iter_var_recall_est = para['N_iter_var_recall_est']

    torch.manual_seed(seed)

    predicted_class = Model_pred(dataloader_test, Predictor, device) #generate y_pred

    res  = torch.empty((0), dtype=torch.float32, device=device)
    res_square  = torch.empty((0), dtype=torch.float32, device=device)

    
    for i in range(N_iter):
        z_pool = torch.randn(z_dim, device=device)# sample z
        ENN_output_list = torch.empty((0), dtype=torch.float32, device=device)
        for (x_batch, label_batch) in dataloader_test:
            #x_batch = x_batch.to(device)
            #label_batch = label_batch.to(device)
            #shifted z outside from here
            fnet_logits = fnet(x_batch, z_pool) 
            #fnet_logits_softmax = torch.nn.Softmax(fnet_logits, dim = 1)
            fnet_logits_probs = torch.nn.functional.softmax(fnet_logits, dim=1)
            ENN_output_list = torch.cat((ENN_output_list,fnet_logits_probs[:,1]),0) 
        #recall est over multiple Gumbel RV
        recall_est_list = torch.empty((0), dtype=torch.float32, device=device)
        for j in range(N_iter_var_recall_est):
            recall_est = Recall(ENN_output_list, predicted_class, tau, seed + j, device).view(1) #use diff seeds for Gumbel
            recall_est_list = torch.cat((recall_est_list, recall_est),0)
        #print("recall_est:", recall_est)
        res = torch.cat((res,torch.mean(recall_est_list).view(1)),0) #append mean of recall over multiple Gumbel
        #print("res:",res)
        res_square = torch.cat((res_square,(recall_est ** 2).view(1)),0)
        
    #print("res_square:", res_square)
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
