"""Subset sampling: the soft b-subset Gumbel relaxation of Xie & Ermon (Algorithm 1 of the paper,
ported from k_subset_sampling.SubsetOperator) and exact hard sampling with the matching
Plackett-Luce (sequential sampling without replacement) log-probability for REINFORCE."""
import numpy as np
import torch

EPSILON = float(np.finfo(np.float32).tiny)


def gumbel_like(x: torch.Tensor, generator=None):
    u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator).clamp_(1e-20, 1.0)
    return -torch.log(-torch.log(u))


def soft_topk(scores: torch.Tensor, k: int, tau: float, noise: torch.Tensor | None = None, generator=None,
              clamp_max: float | None = 1.0):
    """Relaxed k-hot vector(s).  scores: [..., n] logits (log-weights).  Adds Gumbel noise
    (unless `noise` given), then runs the recursive softmax of Xie & Ermon.  Returns [..., n]
    with entries in [0, 1] summing to ~k (exactly k before clamping)."""
    if noise is None:
        noise = gumbel_like(scores, generator)
    s = scores + noise
    khot = torch.zeros_like(s)
    onehot_approx = torch.zeros_like(s)
    eps = torch.tensor(EPSILON, device=s.device, dtype=s.dtype)
    for _ in range(k):
        mask = torch.maximum(1.0 - onehot_approx, eps)
        s = s + torch.log(mask)
        onehot_approx = torch.softmax(s / tau, dim=-1)
        khot = khot + onehot_approx
    if clamp_max is not None:
        khot = khot.clamp(max=clamp_max)
    return khot


def soft_topk_slots(scores: torch.Tensor, k: int, tau: float, noise: torch.Tensor | None = None, generator=None):
    """The k successive relaxed one-hot vectors of soft_topk's recursion, stacked: [..., k, n], each summing to 1 (their sum
    is soft_topk's k-hot vector before clamping).  Used by belief modules that relax a batch as k slots, each a convex
    combination of pool items, instead of fractional item weights."""
    if noise is None:
        noise = gumbel_like(scores, generator)
    s = scores + noise
    onehot_approx = torch.zeros_like(s)
    eps = torch.tensor(EPSILON, device=s.device, dtype=s.dtype)
    slots = []
    for _ in range(k):
        mask = torch.maximum(1.0 - onehot_approx, eps)
        s = s + torch.log(mask)
        onehot_approx = torch.softmax(s / tau, dim=-1)
        slots.append(onehot_approx)
    return torch.stack(slots, -2)


def sample_hard_subset(logits: torch.Tensor, k: int, generator=None):
    """Exact draw from the policy of eq. (39)-(40): Gumbel-top-k.  Returns the ordered indices
    (descending perturbed score == sequential sampling order)."""
    g = gumbel_like(logits, generator)
    return torch.topk(logits + g, k).indices


def ordered_logprob(logits: torch.Tensor, idx: torch.Tensor):
    """log pi^ord(i_1..i_b) = sum_j [ logits[i_j] - logsumexp(logits over items not yet chosen) ]."""
    n = logits.shape[-1]
    mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
    lp = logits.new_zeros(())
    for j in range(idx.numel()):
        avail = logits.masked_fill(mask, float("-inf"))
        lp = lp + logits[idx[j]] - torch.logsumexp(avail, dim=-1)
        mask = mask | torch.nn.functional.one_hot(idx[j], n).bool()  # out-of-place (autograd-safe)
    return lp


def onehot_from_indices(idx: torch.Tensor, n: int, dtype=torch.float32):
    v = torch.zeros(n, dtype=dtype, device=idx.device)
    v[idx] = 1.0
    return v
