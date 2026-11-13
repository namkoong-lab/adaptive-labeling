"""Why was the original Ensemble+ pathwise planner slow, and how much faster is the new one?

Compares, on the same problem (d=5, 10 members, 2x50 MLP, N=120 training points incl. soft-weighted
pool, G=10 rollouts), the cost of ONE pathwise policy-gradient estimate:
  (a) original style: for each rollout, for each member, a Python loop of inner Adam steps through
      torchopt.MetaAdam, then backward through the unrolled trajectory (as in enn_pipeline_regression_1a.py);
  (b) new: all G*M replicas trained at once with batched matmuls and a plain-torch unrolled Adam
      (adaptive_sampling_lib.uq_ensemble), backward once.
Both use 50 inner steps here (the paper's number); the new module is run with 50 and 200 steps.
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torchopt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from adaptive_sampling_lib.data import make_synthetic_gp_problem
from adaptive_sampling_lib.uq_ensemble import EnsemblePlusModule
from adaptive_sampling_lib.utils import pick_device, seed_all

dev = pick_device(int(sys.argv[1]) if len(sys.argv) > 1 else None)
seed_all(0)
prob = make_synthetic_gp_problem(0, 5, n_train=20, n_pool=100, n_anchor=11, eval_per_cluster=20, layout="gmm", center_scale=0.5, cluster_std=0.5).to(dev)
G, M, steps = 10, 10, 50
X = torch.cat([prob.train_x, prob.pool_x]); y = torch.cat([prob.train_y, prob.pool_y])
w = torch.cat([torch.ones(G, 20, device=dev), torch.rand(G, 100, device=dev) * 0.1], 1).requires_grad_(True)


def mlp(d):
    return nn.Sequential(nn.Linear(d, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1))


# ---- (a) original style: per-rollout, per-member torchopt MetaAdam loop ----
priors = [mlp(5).to(dev) for _ in range(M)]
for p_ in priors:
    for q in p_.parameters():
        q.requires_grad_(False)
alpha = 1.0


def original_estimate():
    total = 0.0
    for g in range(G):
        preds = []
        for m in range(M):
            net = mlp(5).to(dev)
            opt = torchopt.MetaAdam(net, lr=0.1, moment_requires_grad=False)
            for _ in range(steps):
                out = net(X) + alpha * priors[m](X)
                loss = (w[g] * (out.squeeze(-1) - y) ** 2).sum() + 0.1 * sum((p ** 2).sum() for p in net.parameters())
                opt.step(loss)  # differentiable update
            preds.append((net(prob.eval_x) + alpha * priors[m](prob.eval_x)).squeeze(-1))
        F = torch.stack(preds)  # [M, N_eval]
        Q = (F ** 2).mean(-1)
        risk = Q.var()
        (risk / G).backward()
        total += risk.item()
    return total / G


def new_estimate(uq):
    st = {"X": prob.train_x, "y": prob.train_y, "params": None}
    draws = uq.rollout_draws(st, prob.eval_x, prob.pool_x, prob.pool_y, w[:, 20:], None, differentiable=True)
    risk = (draws ** 2).mean(-1).var(dim=1).mean()
    risk.backward()
    return risk.item()


for name, fn in [("original (torchopt MetaAdam, python loops, 50 steps)", original_estimate)]:
    w.grad = None
    torch.cuda.synchronize(); t0 = time.time(); fn(); torch.cuda.synchronize()
    print(f"{name}: {time.time() - t0:.1f} s per gradient estimate  (|grad| {w.grad.norm().item():.3e})")
for n_inner in [50, 200]:
    uq = EnsemblePlusModule(5, (50, 50), M, alpha=1.0, n_inner_steps=n_inner, inner_lr=0.1, weight_decay=0.1, device=dev, anchor_to_init=False)
    w.grad = None
    new_estimate(uq)  # warm-up
    w.grad = None
    torch.cuda.synchronize(); t0 = time.time(); new_estimate(uq); torch.cuda.synchronize()
    print(f"new (batched, plain-torch unrolled Adam, {n_inner} steps): {time.time() - t0:.2f} s per gradient estimate  (|grad| {w.grad.norm().item():.3e})")
