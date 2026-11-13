import torch
import gpytorch
import torch.nn as nn

class Naive_Network(nn.Module):
    def __init__(self, output_size=1):
        super(Naive_Network, self).__init__()
        self.output_size = output_size
        self.random_tensor = torch.rand(1000)

    def forward(self, x):
        size = x.size(0)
        device = x.device
        return (self.random_tensor[:size]).unsqueeze(1).to(device)
        #return (torch.sigmoid(torch.mean(x,dim=1))).unsqueeze(1)




# class ConstantValueNetwork(nn.Module):
#     def __init__(self, constant_value=1.0, output_size=1):
#         super(ConstantValueNetwork, self).__init__()
#         # Define the constant value and output size
#         self.constant_value = nn.Parameter(torch.tensor([constant_value]*output_size), requires_grad=False)
#         self.output_size = output_size

#     def forward(self, x):
#         # x is your input tensor. Its value is ignored in this model.
#         # Return a 1-D tensor with the constant value for each item in the batch.
#         batch_size = x.size(0)  # Get the batch size from the input
#         return self.constant_value.expand(batch_size, self.output_size)