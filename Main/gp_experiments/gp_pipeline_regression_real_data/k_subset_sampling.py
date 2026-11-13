import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

EPSILON = np.finfo(np.float32).tiny

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, device, tau=1.0, hard=False):            # k is the number of samples we want, tau is the temperature parameter, hard:denotes if we want to implement the straight through implementation or the simple implementation
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
        self.device=device

    def forward(self, scores):                                # scores take in log(weights) of each sample      # scores: Typical shape: [batch_size,n] or [batch_size,n,1]  where you want to choose k elements for n elements
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k  (we can later modify this to also output S_WRS, we will just need each onehot_approx to be stored seperately - then it will give k soft vectors)
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON], device=self.device))            # we can autodiff through this, there is no issue .
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # will do straight through estimation if training
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res