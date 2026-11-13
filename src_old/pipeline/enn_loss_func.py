import torch
import torch.nn.functional as F

def weighted_nll_loss(log_probs, targets, weights):
    """
    Custom weighted Negative Log Likelihood Loss
    :param log_probs: Log probabilities (output of log-softmax) from the model.   #[N, C] - dim
    :param targets: Target labels.   #[N] - dim
    :param weights: Weights for each sample in the batch.    #[N] - dim
    :return: Weighted NLL loss
    """
    # NLL loss for each sample
    nll_loss = F.nll_loss(log_probs, targets, reduction='none')

    # Apply weights
    weighted_loss = nll_loss * weights

    # Average the weighted losses
    return weighted_loss.mean()

#MIGHT BE USEFUL FOR ABOVE FUNCTION
# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#         # Initialize any parameters or layers

#     def forward(self, output, target):
#         # Define your custom loss computation
#         loss = torch.mean((output - target) ** 2)  # Example: Mean Squared Error
#         return loss

# # Usage example
# criterion = CustomLoss()
# output = model(input)
# target = ...  # Your target tensor
# loss = criterion(output, target)
# loss.backward()