import torch

'''
Code from
https://github.com/namkoong-lab/adaptive_sampling/blob/64d42bd895b0ac4185a57e1a2aaa00b1637ceb26/src/autodiff/enn_pipeline_classification_yuanzhe/active_learning_algos.py

Args:
logits: An array of shape [A, B, C] where 
A is # of ensemble
B is the batch size of data
C  is the number of outputs per data (for classification, this is equal to
number of classes), and A is the number of random samples for each data.
labels: An array of shape [B, 1] where B is the batch size of data.
 
Returns:
  A priority score per example of shape [B,].
'''
def uniform_per_example(logits):
    """Returns uniformly random scores per example."""
    return torch.rand(logits.shape[1])

def variance_per_example(logits):
    """Calculates variance per example."""
    _, data_size, _ = logits.shape
    probs = torch.nn.functional.softmax(logits, dim=-1)
    variances = torch.var(probs, dim=0, unbiased=False).sum(dim=-1)
    assert variances.shape == (data_size,)
    return variances

def entropy_per_example(logits):
    """Calculates entropy per example."""
    _, data_size, num_classes = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.mean(sample_probs, dim=0)
    assert probs.shape == (data_size, num_classes)

    entropies = -1 * torch.sum(probs * torch.log(probs), dim=1)
    assert entropies.shape == (data_size,)

    return entropies

def margin_per_example(logits):
    """Calculates margin between top and second probabilities per example."""
    _, data_size, num_classes = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.mean(sample_probs, dim=0)
    assert probs.shape == (data_size, num_classes)

    sorted_probs, _ = torch.sort(probs, descending=True)
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    assert margins.shape == (data_size,)

    # Return the *negative* margin
    return -margins

def bald_per_example(logits):
    """Calculates BALD mutual information per example."""
    num_enn_samples, data_size, num_classes = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)

    # Function to compute entropy
    def compute_entropy(p):
        return -1 * torch.sum(p * torch.log(p), dim=1)

    # Compute entropy for average probabilities
    mean_probs = torch.mean(sample_probs, dim=0)
    assert mean_probs.shape == (data_size, num_classes)
    mean_entropy = compute_entropy(mean_probs)
    assert mean_entropy.shape == (data_size,)

    # Compute entropy for each sample probabilities
    sample_entropies = torch.stack([compute_entropy(p) for p in sample_probs])
    assert sample_entropies.shape == (num_enn_samples, data_size)

    models_disagreement = mean_entropy - torch.mean(sample_entropies, dim=0)
    assert models_disagreement.shape == (data_size,)
    return models_disagreement



def var_ratios_per_example(logits):
    """Calculates the highest probability per example."""
    _, data_size, num_classes = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.mean(sample_probs, dim=0)
    assert probs.shape == (data_size, num_classes)

    max_probs = torch.max(probs, dim=1).values
    variation_ratio = 1 - max_probs
    assert len(variation_ratio) == data_size

    return variation_ratio

active_learning_dict = {'uniform_per_example':uniform_per_example,
'variance_per_example':variance_per_example,                     
'entropy_per_example':entropy_per_example,
'margin_per_example':margin_per_example,
'bald_per_example':bald_per_example,
'var_ratios_per_example':var_ratios_per_example}
 
def select_samples_active_learning(data, acquisition_size, algo, enn, n_z_samples, z_dim, seed, n_classes = 2):
    '''
    input: data (only has X feature), acquisition_size = batch_size
    algo in ['uniform_per_example', 'variance_per_example', 'entropy_per_example', 'margin_per_example', 'bald_per_example', 'var_ratios_per_example']
    enn is a trained model, n_z_sample is samples used to draw z
    '''
    logits = torch.zeros((n_z_samples,data.shape[0],n_classes)) 
    torch.manual_seed(seed)
    for i in range(n_z_samples):
        z = torch.randn(z_dim)
        logits[i] = enn(data,z)
    candidate_scores = active_learning_dict[algo](logits)
    selected_idxs = torch.argsort(candidate_scores, descending=True)[:acquisition_size]
    acquired_data = data[selected_idxs]
    return acquired_data

