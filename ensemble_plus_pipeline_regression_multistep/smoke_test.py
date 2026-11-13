"""Speed/correctness check of the vectorized Ensemble+ module and the planners on top of it."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_sampling_lib.data import make_synthetic_gp_problem
from adaptive_sampling_lib.planners import PolicyGradientPlanner, select_uncertainty_sequential, select_uncertainty_static
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs
from adaptive_sampling_lib.runner import track_metrics
from adaptive_sampling_lib.targets import MSETarget
from adaptive_sampling_lib.uq_ensemble import EnsemblePlusModule
from adaptive_sampling_lib.utils import pick_device, seed_all

dev = pick_device(int(sys.argv[1]) if len(sys.argv) > 1 else None)
seed_all(0)
prob = make_synthetic_gp_problem(0, 1, n_train=20, n_pool=100, n_anchor=11).to(dev)
print(prob.summary())
print("eval cluster hist:", torch.bincount(prob.eval_cluster).tolist(), "pool:", torch.bincount(prob.pool_cluster).tolist())
pred = train_predictor(prob.train_x, prob.train_y, device=dev)
psi = predictor_outputs(pred, prob.eval_x)
target = MSETarget(psi, 0.01)
uq = EnsemblePlusModule(1, (50, 50), 10, alpha="auto", prior_target_std=0.83, n_inner_steps=50, inner_lr=0.1, weight_decay=0.1,
                        noise_std=0.1, device=dev, calib_x=torch.cat([prob.eval_x, prob.pool_x]))
print("alpha", uq.alpha)
t0 = time.time()
st = uq.condition(prob.train_x, prob.train_y)
torch.cuda.synchronize() if dev.type == "cuda" else None
print(f"condition (50 inner steps, 10 members): {time.time() - t0:.2f}s")
m = track_metrics(uq, st, target, prob.eval_x, prob.eval_y, 10)
print("track:", m)
# fit quality on train
f = uq.posterior_draws(st, prob.train_x)
print("train rmse of member means:", ((f.mean(0) - prob.train_y) ** 2).mean().sqrt().item(), " member spread at train:", f.std(0).mean().item())
fe = uq.posterior_draws(st, prob.eval_x)
print("member spread at eval (by cluster):", [round(fe.std(0)[prob.eval_cluster == c].mean().item(), 3) for c in range(11)])
ybar = uq.sample_pool_labels(st, prob.pool_x, 1)[0]
# rollout speed: G=10 soft weights, differentiable
W = torch.rand(10, prob.pool_x.shape[0], device=dev, requires_grad=True)
t0 = time.time()
draws = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, W, differentiable=True)
Q = target.q(draws)
risk = Q.var(dim=1).mean()
risk.backward()
torch.cuda.synchronize() if dev.type == "cuda" else None
print(f"differentiable rollout G=10 (fwd+bwd): {time.time() - t0:.2f}s  grad norm {W.grad.norm().item():.3e}  risk {risk.item():.4f}")
t0 = time.time()
with torch.no_grad():
    draws = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, (W > 0.9).float(), differentiable=False)
torch.cuda.synchronize() if dev.type == "cuda" else None
print(f"hard rollout G=10: {time.time() - t0:.2f}s")
for est, bl in [("pathwise", "none"), ("reinforce", "mean")]:
    pl = PolicyGradientPlanner(uq, target, 5, n_steps=5, G=10, n_mc=10, lr=0.1, tau=0.1, estimator=est, baseline=bl, verbose=True, log_every=1)
    t0 = time.time()
    idx, hist = pl.select(st, prob.pool_x, prob.eval_x)
    print(est, "selected", idx.tolist(), "clusters", prob.pool_cluster[idx.to(dev)].tolist(), f"{(time.time() - t0) / 5:.2f}s/step")
print("unc static:", prob.pool_cluster[select_uncertainty_static(uq, st, prob.pool_x, 5).to(dev)].tolist())
print("unc seq   :", prob.pool_cluster[select_uncertainty_sequential(uq, st, prob.train_x, prob.train_y, prob.pool_x, 5).to(dev)].tolist())
print("OK")
