import torch
import gpytorch
import matplotlib.pyplot as plt
import torch.nn as nn

num_init_train_samples = 20
num_pool_samples = 5
num_test_samples = 20
input_dim = 1

init_train_x = torch.rand((num_init_train_samples, input_dim))*50.0
test_x_1 = torch.rand((num_test_samples, input_dim))*50.0
test_x_2 = 75.0 + torch.rand((num_test_samples, input_dim))*50.0
test_x_3 = 175.0 + torch.rand((num_test_samples, input_dim))*50.0
test_x = torch.cat([test_x_1,test_x_2,test_x_3])
pool_x_1 = 24 + torch.rand((num_pool_samples, input_dim))*2
pool_x_2 = 99 + torch.rand((num_pool_samples, input_dim))*2
pool_x_3 = 199 + torch.rand((num_pool_samples, input_dim))*2
pool_x = torch.cat([pool_x_1,pool_x_2,pool_x_3])
y = torch.zeros(num_init_train_samples+3*num_pool_samples+3*num_test_samples)

init_train_x_numpy = init_train_x.numpy()
init_train_y = torch.zeros(init_train_x.size(0))
test_x_numpy = test_x.numpy()
test_y = torch.ones(test_x.size(0))
pool_x_numpy = pool_x.numpy()
pool_y = torch.empty(pool_x.size(0)).fill_(0.5)


fig = plt.figure()
plt.scatter(init_train_x_numpy, init_train_y.numpy(), s=20, label='train')
plt.scatter(test_x_numpy, test_y.numpy(), s=20, label='test')
plt.scatter(pool_x_numpy, pool_y.numpy(), s=20, label='pool')

plt.yticks([])  # Hide y-axis ticks
plt.xlabel('X values')
plt.legend()
plt.title('Distribution of X values along a real line')
plt.show()

x = torch.cat([init_train_x,test_x,pool_x])

# Define parameters for the model
mean_constant = 0.0  # Mean of the GP
length_scale = 25.0   # Length scale of the RBF kernel
noise_std = 0.01     # Standard deviation of the noise


mean_module = gpytorch.means.ConstantMean()
base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()


mean_module.constant = mean_constant
base_kernel.base_kernel.lengthscale = length_scale
likelihood.noise_covar.noise = noise_std**2

class CustomizableGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, base_kernel, likelihood):
        super(CustomizableGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = base_kernel
        self.likelihood = likelihood

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


model = CustomizableGPModel(x, y, mean_module, base_kernel, likelihood)

# Sample from the prior for training data
model.eval()
likelihood.eval()
with torch.no_grad():
    prior_dist = likelihood(model(x))
    y_new = prior_dist.sample()

plt.scatter(x[:num_init_train_samples],y_new[:num_init_train_samples], label='train')
plt.scatter(x[num_init_train_samples:num_init_train_samples+3*num_test_samples],y_new[num_init_train_samples:num_init_train_samples+3*num_test_samples], label='test')
plt.scatter(x[num_init_train_samples+3*num_test_samples:],y_new[num_init_train_samples+3*num_test_samples:], label='pool')
plt.ylim(-0.1, 0.1)

class ConstantValueNetwork(nn.Module):
    def __init__(self, constant_value=1.0, output_size=1):
        super(ConstantValueNetwork, self).__init__()
        # Define the constant value and output size
        self.constant_value = nn.Parameter(torch.tensor([constant_value]*output_size), requires_grad=False)
        self.output_size = output_size

    def forward(self, x):
        # x is your input tensor. Its value is ignored in this model.
        # Return a 1-D tensor with the constant value for each item in the batch.
        batch_size = x.size(0)  # Get the batch size from the input
        return self.constant_value.expand(batch_size, self.output_size)

### Adapting L_2 loss for the GP pipeine

def var_l2_loss_estimator(model, test_x, Predictor, device, para):

    N_iter =  100
    seed = 0
    torch.manual_seed(seed)

    latent_posterior = model(test_x)
    latent_posterior_sample = latent_posterior.rsample(sample_shape=torch.Size([N_iter]))
    prediction = Predictor(test_x).squeeze()
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample, prediction))
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+model.likelihood.noise
    print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_variance

def l2_loss(test_x, test_y, Predictor, device):
    prediction = Predictor(test_x).squeeze()
    #print("prediction:", prediction)
    #print("test_y:", test_y)
    diff_square = torch.square(torch.subtract(test_y, prediction))
    #print("diff_square:", diff_square)
    return torch.mean(diff_square)

Predictor = ConstantValueNetwork(constant_value=0.0, output_size=1)

model.set_train_data(inputs=x[:num_init_train_samples], targets=y_new[:num_init_train_samples], strict=False)       ####### CAN ALSO USE TRAINING OVER NLL HERE########

### IMP LINK - https://github.com/cornellius-gp/gpytorch/issues/1409
### IMP LINK - https://docs.gpytorch.ai/en/latest/examples/01_Exact_GPs/Simple_GP_Regression.html
posterior = (model(x))
posterior_mean = posterior.mean
posterior_var = posterior.variance
#print("posterior_var:",posterior_var)

fig = plt.figure()
plt.scatter(x,posterior_mean.detach().numpy())
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()-2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()+2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.ylim(-2, 2)

var_l2_loss_estimator(model, test_x, Predictor, None, None)

l_2_loss_actual = l2_loss(test_x, y_new[num_init_train_samples:num_init_train_samples+3*num_test_samples], Predictor, None)
print("l_2_loss_actual:", l_2_loss_actual)

new_train_x = torch.cat([x[:num_init_train_samples],x[-2:]])
new_train_y = torch.cat([y_new[:num_init_train_samples],y_new[-2:]])

model.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)       ####### CAN ALSO USE TRAINING OVER NLL HERE########

posterior = likelihood(model(x))
posterior_mean = posterior.mean
posterior_var = posterior.variance


plt.scatter(x,posterior_mean.detach().numpy())
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()-2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()+2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.ylim(-2, 2)
var_l2_loss_estimator(model, test_x, Predictor, None, None)

new_train_x = torch.cat([x[:num_init_train_samples],x[num_init_train_samples+num_test_samples*3+num_pool_samples+1:num_init_train_samples+num_test_samples*3+num_pool_samples*2+2],x[-1:]])
new_train_y = torch.cat([y_new[:num_init_train_samples],y[num_init_train_samples+num_test_samples*3+num_pool_samples+1:num_init_train_samples+num_test_samples*3+num_pool_samples*2+2],y_new[-1:]])

model.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)       ####### CAN ALSO USE TRAINING OVER NLL HERE########

posterior = likelihood(model(x))
posterior_mean = posterior.mean
posterior_var = posterior.variance

fig = plt.figure()
plt.scatter(x,posterior_mean.detach().numpy())
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()-2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.scatter(x.squeeze(),posterior_mean.detach().numpy()+2*torch.sqrt(posterior_var).detach().numpy(),alpha=0.2)
plt.ylim(-2, 2)
var_l2_loss_estimator(model, test_x, Predictor, None, None)

class RBFKernel(nn.Module):
    def __init__(self, length_scale= 0.6931471824645996, output_scale = 0.6931471824645996):
        super(RBFKernel, self).__init__()
        self.length_scale = length_scale
        self.output_scale = output_scale
    def forward(self, x1, x2):
        dist_matrix = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0)**2
        return self.output_scale*torch.exp(-0.5 * dist_matrix / self.length_scale**2)

class GaussianProcessCholesky(nn.Module):
    def __init__(self, kernel):
        super(GaussianProcessCholesky, self).__init__()
        self.kernel = kernel

    def forward(self, x_train, y_train, w_train, x_test, noise=1e-4):

        # Apply weights only to non-diagonal elements

        K = self.kernel(x_train, x_train) + noise * torch.eye(x_train.size(0)) + 1e-6 * torch.eye(x_train.size(0))
        non_diag_mask = 1 - torch.eye(K.size(-2), K.size(-1))
        weight_matrix = w_train.unsqueeze(-1) * w_train.unsqueeze(-2)
        weighted_K =  K * (non_diag_mask * weight_matrix + (1 - non_diag_mask))



        K_s = self.kernel(x_train, x_test)
        weighted_K_s = torch.diag(w_train)@K_s

        K_ss = self.kernel(x_test, x_test) + 1e-6 * torch.eye(x_test.size(0))

        L = torch.linalg.cholesky(weighted_K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        mu = weighted_K_s.t().matmul(alpha).squeeze(-1)

        v = torch.linalg.solve(L, weighted_K_s)
        cov = K_ss - v.t().matmul(v)

        return mu, cov

import torch

def sample_multivariate_normal(mu, cov, n_samples):
    """
    Sample from a multivariate normal distribution using the reparameterization trick.

    Parameters:
    - mu (torch.Tensor): The mean vector of the distribution.    1-D dimension [D]
    - cov (torch.Tensor): The covariance matrix of the distribution.  2-D dimension [D,D]
    - n_samples (int): The number of samples to generate.

    Returns:
    - torch.Tensor: Samples from the multivariate normal distribution.
    """
    # Ensure mu and cov are tensors
    #mu = torch.tensor(mu, dtype=torch.float32)
    #cov = torch.tensor(cov, dtype=torch.float32)

    # Cholesky decomposition of the covariance matrix
    L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(cov.size(0)))

    #L = torch.linalg.cholesky(cov + 1e-8 * torch.eye(cov.size(0)))

    # Sample Z from a standard normal distribution
    Z = torch.randn(n_samples, mu.size(0))           # Z: [n_samples, D]

    # Transform Z to obtain samples from the target distribution
    samples = mu + Z @ L.T

    return samples    #[n_samples, D]

### Adapting L_2 loss for the GP pipeine

def var_l2_loss_custom_gp_estimator(mu, cov, noise, test_x, Predictor, device, para):


    N_iter =  100
    seed = 0
    torch.manual_seed(seed)

    latent_posterior_sample = sample_multivariate_normal(mu, cov, N_iter)
    prediction = Predictor(test_x).squeeze()
    L_2_loss_each_point = torch.square(torch.subtract(latent_posterior_sample, prediction))
    L_2_loss_each_f = torch.mean(L_2_loss_each_point, dim=1)
    L_2_loss_variance = torch.var(L_2_loss_each_f)
    print("L_2_loss_variance:",L_2_loss_variance)

    L_2_loss_mean = torch.mean(L_2_loss_each_f)+noise
    print("L_2_loss_mean:", L_2_loss_mean)

    return L_2_loss_variance

x_train = x[:num_init_train_samples]
x_pool_1 = x[num_init_train_samples+num_test_samples*3:num_init_train_samples+num_test_samples*3+num_pool_samples*3-2]
x_pool_2 = x[num_init_train_samples+num_test_samples*3+num_pool_samples*3-2:]

y_train = y_new[:num_init_train_samples]
y_pool_1 = y_new[num_init_train_samples+num_test_samples*3:num_init_train_samples+num_test_samples*3+num_pool_samples*3-2]
y_pool_2 = y_new[num_init_train_samples+num_test_samples*3+num_pool_samples*3-2:]


x_gp = torch.cat([x_train,x_pool_1,x_pool_2], dim=0)
y_gp = torch.cat([y_train,y_pool_1,y_pool_2], dim=0)

w_train = torch.ones(x_train.size(0), requires_grad = True)
w_pool_1 = torch.zeros(x_pool_1.size(0), requires_grad = True)
w_pool_2 = torch.zeros(x_pool_2.size(0), requires_grad = True)
w_gp = torch.cat([w_train,w_pool_1,w_pool_2])



kernel = RBFKernel(length_scale=25.0, output_scale = 0.6931471824645996)
gp = GaussianProcessCholesky(kernel=kernel)
noise = 1e-4
# Prediction
mu2, cov2 = gp(x_gp, y_gp, w_gp, test_x, noise)

var_l2_loss_custom_gp_estimator(mu2, cov2, 1e-4, test_x, Predictor, None, None)

plt.scatter(test_x,mu2.detach().numpy())
plt.scatter(test_x.squeeze(),mu2.detach().numpy()-2*torch.sqrt(torch.diag(cov2)).detach().numpy(),alpha=0.2)
plt.scatter(test_x.squeeze(),mu2.detach().numpy()+2*torch.sqrt(torch.diag(cov2)).detach().numpy(),alpha=0.2)
plt.ylim(-2, 2)

x_train = x[:num_init_train_samples]
x_pool_1 = x[num_init_train_samples+num_test_samples*3:num_init_train_samples+num_test_samples*3+num_pool_samples*3-2]
x_pool_2 = x[num_init_train_samples+num_test_samples*3+num_pool_samples*3-2:]

y_train = y_new[:num_init_train_samples]
y_pool_1 = y_new[num_init_train_samples+num_test_samples*3:num_init_train_samples+num_test_samples*3+num_pool_samples*3-2]
y_pool_2 = y_new[num_init_train_samples+num_test_samples*3+num_pool_samples*3-2:]


x_gp = torch.cat([x_train,x_pool_1,x_pool_2], dim=0)
y_gp = torch.cat([y_train,y_pool_1,y_pool_2], dim=0)

w_train = torch.ones(x_train.size(0), requires_grad = True)
w_pool_1 = torch.zeros(x_pool_1.size(0), requires_grad = True)
w_pool_2 = torch.ones(x_pool_2.size(0), requires_grad = True)
w_gp = torch.cat([w_train,w_pool_1,w_pool_2])



kernel = RBFKernel(length_scale=25.0)
gp = GaussianProcessCholesky(kernel=kernel)

# Prediction
mu, cov = gp(x_gp, y_gp, w_gp, test_x)

var_loss = var_l2_loss_custom_gp_estimator(mu, cov, 1e-4, test_x, Predictor, None, None)

var_loss.backward()

plt.scatter(test_x,mu.detach().numpy())
plt.scatter(test_x.squeeze(),mu.detach().numpy()-2*torch.sqrt(torch.diag(cov)).detach().numpy(),alpha=0.2)
plt.scatter(test_x.squeeze(),mu.detach().numpy()+2*torch.sqrt(torch.diag(cov)).detach().numpy(),alpha=0.2)
plt.ylim(-2, 2)

#### Advanced version for training as well


class RBFKernelAdvanced(nn.Module):
    def __init__(self, length_scale_init=0.6931471824645996, variance_init=0.6931471824645996):
        super(RBFKernelAdvanced, self).__init__()
        self.raw_length_scale = nn.Parameter(torch.tensor([length_scale_init], dtype=torch.float))
        self.raw_variance = nn.Parameter(torch.tensor([variance_init], dtype=torch.float))

        self.softplus = nn.Softplus()

    def forward(self, x1, x2):
        length_scale = self.softplus(self.raw_length_scale)
        variance = self.softplus(self.raw_variance)
        #length_scale = self.raw_length_scale
        #variance = self.raw_variance
        #sqdist = torch.cdist(x1, x2) ** 2
        dist_matrix = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0)**2
        return variance * torch.exp(-0.5  * dist_matrix / length_scale ** 2)


class GaussianProcessCholeskyAdvanced(nn.Module):
    def __init__(self, length_scale_init=0.6931471824645996, variance_init=0.6931471824645996, noise_var_init=0.1):
        super(GaussianProcessCholeskyAdvanced, self).__init__()
        self.rbf_kernel = RBFKernelAdvanced(length_scale_init=length_scale_init, variance_init=variance_init)
        self.raw_noise_var = nn.Parameter(torch.tensor([noise_var_init], dtype=torch.float))

        self.softplus = nn.Softplus()

    def forward(self, x_train, y_train, w_train, x_test):

        # Apply weights only to non-diagonal elements

        noise_var = self.softplus(self.raw_noise_var)

        K = self.kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + 1e-6 * torch.eye(x_train.size(0), device=x_train.device)
        non_diag_mask = 1 - torch.eye(K.size(-2), K.size(-1), device=x_train.device)
        weight_matrix = w_train.unsqueeze(-1) * w_train.unsqueeze(-2)
        weighted_K =  K * (non_diag_mask * weight_matrix + (1 - non_diag_mask))



        K_s = self.kernel(x_train, x_test)
        weighted_K_s = torch.diag(w_train)@K_s

        K_ss = self.kernel(x_test, x_test) + 1e-6 * torch.eye(x_test.size(0), device=x_test.device)

        L = torch.linalg.cholesky(weighted_K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        mu = weighted_K_s.t().matmul(alpha).squeeze(-1)

        v = torch.linalg.solve(L, weighted_K_s)
        cov = K_ss - v.t().matmul(v)

        return mu, cov

    def nll(self, x_train, y_train):

        noise_var = self.softplus(self.raw_noise_var)
        #noise_var = self.raw_noise_var

        K = self.rbf_kernel(x_train, x_train) + noise_var * torch.eye(x_train.size(0), device=x_train.device) + 1e-6 * torch.eye(x_train.size(0), device=x_train.device)
        #print(K)
        #print("K:", K)
        L = torch.linalg.cholesky(K)
        #print("L:", L)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
        #print("alpha:", alpha)
        nll = 0.5 * y_train.dot(alpha.flatten())
        #print("nll:", nll)
        nll += torch.log(torch.diag(L)).sum()
        #print("nll:", nll)
        #print("0.5 * len(x_train) * torch.log(2 * torch.pi):", 0.5 * len(x_train) * torch.log(torch.tensor(2 * torch.pi, device=nll.device)))
        nll += 0.5 * len(x_train) * torch.log(torch.tensor(2 * torch.pi, device=nll.device))
        print("nll:", nll)
        return nll

gp_model = GaussianProcessCholeskyAdvanced(length_scale_init=25.0, variance_init=0.6931471824645996, noise_var_init=0.0001)
loss = gp_model.nll(x_train, y_train)

def train_gp_model(gp_model, x_train, y_train, optimizer, num_epochs=50):
    gp_model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear previous gradients
        loss = gp_model.nll(x_train, y_train)  # Compute the loss (NLL)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        if (epoch + 1) % 10 == 0:
            print_model_parameters(gp_model)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

gp_model = GaussianProcessCholeskyAdvanced(length_scale_init=25.0, variance_init=0.69, noise_var_init=0.0001)
optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.01, weight_decay = 0.01)
print_model_parameters(gp_model)

x_train = x[:num_init_train_samples]
y_train = y_new[:num_init_train_samples]


# Train the model
train_gp_model(gp_model, x_train, y_train, optimizer, num_epochs=10000)