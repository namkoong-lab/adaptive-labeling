# -*- coding: utf-8 -*-
"""active_learning_algos.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r6er7Zr2TdPP2QAv-_a4-7Nv3H3C9Tr1
"""

import torch
import typing as tp

"""Calculates a priority score based on logits, labels, and a random key.

    Args:
      logits: An array of shape [A, B, C] where B is the batch size of data, C
        is the number of outputs per data (for classification, this is equal to
        number of classes), and A is the number of random samples for each data.
      labels: An array of shape [B, 1] where B is the batch size of data.
      key: A random key.

    Returns:
      A priority score per example of shape [B,].
    """



def uniform_per_example(logits, labels):
    """Returns uniformly random scores per example."""
    del logits  # logits are not used in this function
    labels = labels.squeeze()
    return torch.rand(labels.shape)


def variance_per_example(logits):
    """Calculates variance per example."""
    _, data_size, _ = logits.shape
    probs = torch.nn.functional.softmax(logits, dim=-1)
    variances = torch.var(probs, dim=0, unbiased=False).sum(dim=-1)
    assert variances.shape == (data_size,)
    return variances



def nll_per_example(logits, labels):
    """Calculates negative log-likelihood (NLL) per example."""
    _, data_size, _ = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.mean(sample_probs, dim=0)

    # Penalize with log loss
    labels = labels.to(torch.int64)  # Ensure labels are integers
    labels = labels.squeeze()
    true_probs = probs[torch.arange(data_size), labels]
    losses = -torch.log(true_probs)
    assert losses.shape == (data_size,)
    return losses



def joint_nll_per_example(logits, labels):
    """Calculates joint negative log-likelihood (NLL) per example."""
    num_enn_samples, data_size, _ = logits.shape
    sample_probs = torch.nn.functional.softmax(logits, dim=-1)

    # Penalize with log loss
    labels = labels.to(torch.int64)  # Ensure labels are integers
    labels = labels.squeeze()
    true_probs = sample_probs[:, torch.arange(data_size), labels]
    tau = 10
    repeated_lls = tau * torch.log(true_probs)
    assert repeated_lls.shape == (num_enn_samples, data_size)

    # Take average of joint lls over num_enn_samples
    joint_lls = torch.mean(repeated_lls, dim=0)
    assert joint_lls.shape == (data_size,)

    return -1 * joint_lls



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




def make_ucb_per_example(ucb_factor: float = 1., class_values: tp.Optional[torch.Tensor] = None):
    """Creates a UCB-style priority metric."""

    def compute_ucb(logits, labels, key=None):
        del labels, key
        _, data_size, num_classes = logits.shape

        # Either use class values or default to just the first class
        scale_values = class_values
        if scale_values is None:
            scale_values = torch.zeros(num_classes)
            scale_values[0] = 1

        probs = torch.nn.functional.softmax(logits, dim=-1)
        value = torch.einsum('zbc,c->zb', probs, scale_values)
        mean_values = torch.mean(value, dim=0)
        std_values = torch.std(value, dim=0, unbiased=False)
        ucb_value = mean_values + ucb_factor * std_values
        assert ucb_value.shape == (data_size,)
        return ucb_value

    return compute_ucb





def make_scaled_mean_per_example(class_values: tp.Optional[torch.Tensor] = None):
    """Creates a priority metric based on mean probs scaled by class_values."""

    def compute_scaled_mean(logits, labels, key=None):
        del labels, key
        _, data_size, num_classes = logits.shape

        # Either use class values or default to just the first class
        scale_values = class_values
        if scale_values is None:
            scale_values = torch.zeros(num_classes)
            scale_values[0] = 1

        probs = torch.nn.functional.softmax(logits, dim=-1)
        values = torch.einsum('zbc,c->zb', probs, scale_values)
        mean_values = torch.mean(values, dim=0)
        assert mean_values.shape == (data_size,)
        return mean_values

    return compute_scaled_mean




def make_scaled_std_per_example(class_values: tp.Optional[torch.Tensor] = None):
    """Creates a priority metric based on std of probs scaled by class_values."""

    def compute_scaled_std(logits, labels, key=None):
        del labels, key
        _, data_size, num_classes = logits.shape

        # Either use class values or default to just the first class
        scale_values = class_values
        if scale_values is None:
            scale_values = torch.zeros(num_classes)
            scale_values[0] = 1

        probs = torch.nn.functional.softmax(logits, dim=-1)
        values = torch.einsum('zbc,c->zb', probs, scale_values)
        std_values = torch.std(values, axis=0, unbiased=False)
        assert std_values.shape == (data_size,)
        return std_values

    return compute_scaled_std

batch = [x:y]
acquisition_size = 10

for i in range(n_iter):
  z = torch.randn(z_dim)
  logits = enn(x,z)

labels = y
if labels.ndim == 1:
    labels = labels.unsqueeze(1)

candidate_scores = per_example_priority(logits, labels)

pool_size = len(y)
acquisition_size = min(acquisition_size, pool_size)

selected_idxs = torch.argsort(candidate_scores, descending=True)[:acquisition_size]
acquired_data = {k: v[selected_idxs] for k, v in batch.items()}