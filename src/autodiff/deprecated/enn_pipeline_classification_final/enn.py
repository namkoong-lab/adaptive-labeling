import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

#METHODOLOGY 1
# An implementation that combines basenet, learnable epinet, prior epinet

# shape of x in [batch_size,x_dim], z is [z_dim]
# assuming input is always included and output is never included
# hidden layers and exposed layers same number of entries



class basenet_with_learnable_epinet_and_ensemble_prior(nn.Module):
    def __init__(self, input_size, basenet_hidden_sizes, n_classes, exposed_layers, z_dim, learnable_epinet_hiddens, hidden_sizes_prior, seed_base, seed_learnable_epinet, seed_prior_epinet, alpha):
        super(basenet_with_learnable_epinet_and_ensemble_prior, self).__init__()


        self.z_dim = z_dim
        self.n_classes = n_classes
        self.num_ensemble = z_dim
        self.alpha = alpha


        # Create a list of all sizes (input + hidden + output)
        basenet_all_sizes = [input_size] + basenet_hidden_sizes + [n_classes]

        self.basenet_all_sizes = basenet_all_sizes
        exposed_layers = [True]+exposed_layers+[False]     # assuming input is always included and output is never included

        self.exposed_layers = exposed_layers

        torch.manual_seed(seed_base)
        # Dynamically create layers
        self.basenet_layers = nn.ModuleList()
        for i in range(len(basenet_all_sizes) - 1):
            self.basenet_layers.append(nn.Linear(basenet_all_sizes[i], basenet_all_sizes[i + 1]))


        sum_input_base_epi = sum(basenet_all_size for basenet_all_size, exposed_layer in zip(basenet_all_sizes, exposed_layers) if exposed_layer)

        learnable_epinet_all_sizes = [sum_input_base_epi+z_dim]    + learnable_epinet_hiddens + [n_classes*z_dim]

        self.learnable_epinet_all_sizes = learnable_epinet_all_sizes

        torch.manual_seed(seed_learnable_epinet)
        self.learnable_epinet_layers = nn.ModuleList()
        for j in range(len(learnable_epinet_all_sizes) - 1):
            self.learnable_epinet_layers.append(nn.Linear(learnable_epinet_all_sizes[j], learnable_epinet_all_sizes[j + 1]))




        torch.manual_seed(seed_prior_epinet)
        self.ensemble = nn.ModuleList()
        for _ in range(self.num_ensemble):
            layers = []
            all_sizes_prior = [sum_input_base_epi] + hidden_sizes_prior + [n_classes]
            for i in range(len(all_sizes_prior) - 1):
                layer = nn.Linear(all_sizes_prior[i], all_sizes_prior[i + 1])


                # Initialize weights and biases here
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

                layers.append(layer)
                if i < len(all_sizes_prior) - 2:
                    layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            # Freeze the parameters of this MLP
            for param in mlp.parameters():
                param.requires_grad = False

            self.ensemble.append(mlp)







    def forward(self, x, z):
        hidden_outputs = []
        #concatenate_hidden = x   #assuming x is always input


        for i, (basenet_layer, flag) in enumerate(zip(self.basenet_layers, self.exposed_layers)):
            if flag:
                hidden_outputs.append(x)


            x = basenet_layer(x)

            if i < len(self.basenet_layers) - 1:  # Apply activation function except for the output layer
                x = torch.relu(x)


            #if i>0 and flag:
                #concatenate_hidden = torch.cat(x,concatenate_hidden, dim=1)

        concatenate_hidden = torch.cat(hidden_outputs, dim=1)

        detached_concatenate_hidden = concatenate_hidden.detach()                    ###-------NOT SURE IF BACKPROP WILL WORK PROPERLY THROUGH THIS

        detached_concatenate_hidden_to_prior = concatenate_hidden.detach()
        ###-------NOT SURE IF BACKPROP WILL WORK PROPERLY THROUGH THIS - should we clone and detach


        z_repeated = z.unsqueeze(0).repeat(detached_concatenate_hidden.size(0), 1)

        combined_output = torch.cat([detached_concatenate_hidden,z_repeated], dim=1)




        for j, learnable_epinet_layer in enumerate(self.learnable_epinet_layers):
            combined_output = learnable_epinet_layer(combined_output)

            if j < len(self.learnable_epinet_layers) - 1:  # Apply activation function except for the output layer
                combined_output = torch.relu(combined_output)

        #reshaped_output = combined_output_learnable.view(inputs.shape[0], self.num_classes, self.z_dim)
        reshaped_epinet_output = torch.reshape(combined_output, (combined_output.shape[0], self.n_classes, self.z_dim))

        epinet_output = torch.matmul(reshaped_epinet_output, z)


        outputs_prior = [mlp(detached_concatenate_hidden_to_prior) for mlp in self.ensemble]

        outputs_prior_tensor = torch.stack(outputs_prior, dim=0)

        prior_output = torch.einsum('nbo,n->bo', outputs_prior_tensor, z)

        final_output =  x + epinet_output + self.alpha* prior_output




        return final_output



