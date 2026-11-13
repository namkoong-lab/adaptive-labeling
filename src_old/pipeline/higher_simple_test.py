import torch
import torch.nn as nn
import torch.optim as optim
import higher

class SingleLayerNet(nn.Module):
    def __init__(self, input_size, output_size, seed):
        torch.manual_seed(seed)
        super(SingleLayerNet, self).__init__()
        # Define a single linear layer
        self.linear = nn.Linear(input_size, output_size, bias = False)

    def forward(self, x):
        # Pass the input through the linear layer
        return self.linear(torch.tensor([[1.0]]))

def loss(NN_output, NN_input):
    return ((NN_output-NN_input)**2)

def meta_loss(input):
    return 1.0*(2.0-input)**2

def experiment():     #what are the arguments



    NN_input = torch.tensor([[1.0]], requires_grad=True)

    meta_opt = optim.SGD([NN_input], lr=0.1)


    NN =   SingleLayerNet(1, 1, 0)
    #print(NN)
    for name, param in NN.named_parameters():
        print(f"{name}: {param}")




    for epoch in range(30):
        train(NN_input, meta_opt, NN)
        print("NN_input:",NN_input)

def train(NN_input, meta_opt, NN):
  NN.train()
  NN_opt = torch.optim.SGD(NN.parameters(), lr=0.1)                      # https://higher.readthedocs.io/en/latest/toplevel.html
                                                                       #check what does copy_initial_weights do  # how to give initial training weights to ENN -- this is resolved , if we use same instance of the model everywhere - weights get stored
  meta_opt.zero_grad()
  with higher.innerloop_ctx(NN, NN_opt, copy_initial_weights=False) as (fnet, diffopt):    #copy_initial_weights – if true, the weights of the patched module are copied to form the initial weights of the patched module, and thus are not part of the gradient tape when unrolling the patched module. If this is set to False, the actual module weights will be the initial weights of the patched module.  Similar, to clone and stuff

    for _ in range(30):

      NN_output = fnet(NN_input)
      #print(NN_output)
      loss_iter = loss(NN_output, NN_input)
      print("loss_iter:",loss_iter)
      diffopt.step(loss_iter)     #see_zer_grad
      for name, param in fnet.named_parameters():
        print(f"{name}: {param}")


    meta_loss_iter = meta_loss(fnet(torch.tensor([[1.0]])))      #see how to define a custom loss function   #see where does this calculation for meta_loss happens that is it outside the innerloop_ctx or within it
    print("meta_loss_iter:",meta_loss_iter)
    meta_loss_iter.backward()

  meta_opt.step()

experiment()