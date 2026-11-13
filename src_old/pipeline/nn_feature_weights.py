import torch.nn as nn

class NN_feature_weights(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NN_feature_weights, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# EXAMPLE
# input_size = 2
# net = NN_feature_weights(input_size,[5,5], 1)