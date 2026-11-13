"""Quick correctness/speed check of the library on the 1-D paper setting (and one higher dim).
Checks: (i) weighted GP with binary weights == subset GP; (ii) pathwise gradient is finite and
decreases the rollout risk over a few steps; (iii) REINFORCE step runs; (iv) baselines run."""
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
from adaptive_sampling_lib.uq_gp import GPModule
from adaptive_sampling_lib.utils import pick_device, seed_all

dev = pick_device(int(sys.argv[1]) if len(sys.argv) > 1 else None)
for d in [1, 20]:
    seed_all(0)
    prob = make_synthetic_gp_problem(0, d).to(dev)
    print(prob.summary())
    print("eval cluster hist:", torch.bincount(prob.eval_cluster).tolist()[:6], "... pool:", torch.bincount(prob.pool_cluster).tolist()[:6])
    pred = train_predictor(prob.train_x, prob.train_y, device=dev)
    psi = predictor_outputs(pred, prob.eval_x)
    target = MSETarget(psi, 0.01)
    uq = GPModule(1.0, 0.69, 0.01, device=dev)
    st = uq.condition(prob.train_x, prob.train_y)
    t0 = time.time()
    m = track_metrics(uq, st, target, prob.eval_x, prob.eval_y, 1000)
    print("track:", m, f"{time.time() - t0:.2f}s")
    # (i) weighted == subset
    ybar = uq.sample_pool_labels(st, prob.pool_x, 1)[0]
    idx = torch.tensor([3, 77, 150, 260, 499])
    W = torch.zeros(1, prob.pool_x.shape[0], device=dev)
    W[0, idx] = 1.0
    g = torch.Generator(device=dev).manual_seed(1)
    a = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, W, 50, differentiable=False, generator=g)
    g = torch.Generator(device=dev).manual_seed(1)
    b = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, W, 50, differentiable=True, generator=g)
    print("binary-weight vs subset max abs diff:", (a - b).abs().max().item())
    # (ii) pathwise
    for est, bl in [("pathwise", "none"), ("reinforce", "mean")]:
        pl = PolicyGradientPlanner(uq, target, 5, n_steps=5, G=10, n_mc=100, lr=0.1, tau=0.1, estimator=est, baseline=bl, verbose=True, log_every=1)
        t0 = time.time()
        idx, hist = pl.select(st, prob.pool_x, prob.eval_x)
        print(est, "selected", idx.tolist(), "clusters", prob.pool_cluster[idx.to(dev)].tolist(), f"{time.time() - t0:.1f}s ({(time.time() - t0) / 5:.2f}s/step)")
    print("unc static:", prob.pool_cluster[select_uncertainty_static(uq, st, prob.pool_x, 5).to(dev)].tolist())
    print("unc seq   :", prob.pool_cluster[select_uncertainty_sequential(uq, st, prob.train_x, prob.train_y, prob.pool_x, 5).to(dev)].tolist())
print("OK")
