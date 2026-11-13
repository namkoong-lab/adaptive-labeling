"""Correctness checks for the TabPFN UQ module on the 1-D synthetic GP problem.
1) raw forward vs. the sklearn TabPFNRegressor (sanity), 2) binary weights == subset context,
3) gradient w.r.t. a context weight (autograd vs finite differences), 4) posterior draws and the
value of labelling heavy vs light clusters, 5) timings."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_sampling_lib.data import make_synthetic_gp_problem
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs
from adaptive_sampling_lib.targets import MSETarget, risk_and_estimate
from adaptive_sampling_lib.uq_tabpfn import TabPFNModule
from adaptive_sampling_lib.utils import pick_device, seed_all

dev = pick_device(int(sys.argv[1]) if len(sys.argv) > 1 else None)
seed_all(0)
prob = make_synthetic_gp_problem(0, 1).to(dev)
pred = train_predictor(prob.train_x, prob.train_y, device=dev)
psi = predictor_outputs(pred, prob.eval_x)
target = MSETarget(psi, 0.0)
t0 = time.time()
uq = TabPFNModule("regression", device=dev, n_draws=16, n_chunks=4, autocast=False)
print(f"loaded TabPFN v2.5 regressor in {time.time() - t0:.1f}s; thinking rows={uq.n_thinking}; n_buckets={uq.bardist.num_bars}")
st = uq.condition(prob.train_x, prob.train_y)

# 1) raw predictive vs sklearn wrapper
mean, std = uq.predictive(st, prob.eval_x)
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, device=str(dev), n_estimators=1)
reg.fit(prob.train_x.cpu().numpy(), prob.train_y.cpu().numpy())
sk_mean = reg.predict(prob.eval_x.cpu().numpy())
print("raw-vs-sklearn mean corr:", np.corrcoef(mean.cpu().numpy(), sk_mean)[0, 1], " rmse raw:", float(((mean.cpu() - prob.eval_y.cpu()) ** 2).mean().sqrt()),
      " rmse sklearn:", float(np.sqrt(((sk_mean - prob.eval_y.cpu().numpy()) ** 2).mean())))
print("predictive std by eval cluster (0, 1..3, 40):", [round(std[prob.eval_cluster == c].mean().item(), 3) for c in [0, 1, 2, 3, 40]])

# 2) binary weights == subset
idx = torch.tensor([3, 77, 150, 260, 499], device=dev)
ctx_x = torch.cat([st["X"], prob.pool_x[idx]], 0)
ctx_y = torch.cat([st["y"], uq._znorm(prob.pool_y[idx], st)], 0)
l_sub = uq.forward_raw(ctx_x, ctx_y, prob.eval_x[:50])
W = torch.zeros(prob.pool_x.shape[0], 1, device=dev)
W[idx] = 1.0
l_w = uq.forward_raw(torch.cat([st["X"], prob.pool_x], 0), torch.cat([st["y"], uq._znorm(prob.pool_y, st)], 0), prob.eval_x[:50],
                     ctx_w=torch.cat([torch.ones(st["X"].shape[0], 1, device=dev), W], 0))
m_sub, m_w = uq.bardist.mean(l_sub), uq.bardist.mean(l_w)
print("binary-weight vs subset: max|logit diff| =", (l_sub - l_w).abs().max().item(), " max|mean diff| =", (m_sub - m_w).abs().max().item(),
      " (mean scale ", m_sub.abs().mean().item(), ")")

# 3) gradient check w.r.t. one weight
w = torch.cat([torch.ones(st["X"].shape[0], 1, device=dev), torch.full((prob.pool_x.shape[0], 1), 0.3, device=dev)], 0).requires_grad_(True)
cx = torch.cat([st["X"], prob.pool_x], 0)
cy = torch.cat([st["y"], uq._znorm(prob.pool_y, st)], 0)
qx = prob.eval_x[:40]


def f(wv):
    lg = uq.forward_raw(cx, cy, qx, ctx_w=wv, grad=True)
    return uq.bardist.mean(lg).pow(2).mean()


val = f(w)
val.backward()
j = st["X"].shape[0] + 10
eps = 1e-2
with torch.no_grad():
    wp = w.detach().clone(); wp[j] += eps
    wm = w.detach().clone(); wm[j] -= eps
    fd = (f(wp) - f(wm)).item() / (2 * eps)
print(f"grad check item {j}: autograd={w.grad[j].item():.5e} finite-diff={fd:.5e}; |grad| nonzero entries={int((w.grad.abs() > 0).sum())}")

# 4) posterior draws & value of heavy vs light clusters
g = torch.Generator(device=dev).manual_seed(0)
t0 = time.time()
draws = uq.posterior_draws(st, prob.eval_x, 32, g)
print(f"posterior draws {tuple(draws.shape)} in {time.time() - t0:.2f}s")
var0, est0 = risk_and_estimate(target.q(draws))
print(f"risk(train only): var={var0.item():.4f} est={est0.item():.4f} actual={target.true_value(prob.eval_y):.4f}")
print("draw std by eval cluster (0,1,2,3,40):", [round(draws.std(0)[prob.eval_cluster == c].mean().item(), 3) for c in [0, 1, 2, 3, 40]])
ybar = uq.sample_pool_labels(st, prob.pool_x, 1, g)[0]
print("imagined pool labels: corr with true", float(np.corrcoef(ybar.cpu().numpy(), prob.pool_y.cpu().numpy())[0, 1]))
for name, clusters in [("cluster0", [0]), ("heavy 1-5", [1, 2, 3, 4, 5]), ("light 40-44", [40, 41, 42, 43, 44]), ("heavy 10 x5", [10])]:
    sel = torch.cat([torch.nonzero(prob.pool_cluster == c).squeeze(1)[: (5 if len(clusters) == 1 else 1)] for c in clusters])[:5]
    Wg = torch.zeros(1, prob.pool_x.shape[0], device=dev)
    Wg[0, sel] = 1.0
    t0 = time.time()
    d = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, Wg, 32, differentiable=False, generator=g)
    var, est = risk_and_estimate(target.q(d))
    print(f"  add {name:12s} -> risk var={var[0].item():.4f} ({time.time() - t0:.2f}s)")
# soft rollout, differentiable, G=2
Wsoft = torch.rand(2, prob.pool_x.shape[0], device=dev) * 0.05
Wsoft[0, sel] = 1.0
Wsoft.requires_grad_(True)
t0 = time.time()
d = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, Wsoft, 8, differentiable=True, generator=g)
risk = target.q(d).var(dim=1).mean()
risk.backward()
torch.cuda.synchronize()
print(f"differentiable soft rollout G=2,S=8, all 500 pool rows: {time.time() - t0:.2f}s  risk={risk.item():.4f} grad norm={Wsoft.grad.norm().item():.3e}  mem={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
uq.soft_topm = 50
Wsoft.grad = None
t0 = time.time()
d = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, Wsoft, 8, differentiable=True, generator=g)
risk = target.q(d).var(dim=1).mean()
risk.backward()
torch.cuda.synchronize()
print(f"differentiable soft rollout top-m=50: {time.time() - t0:.2f}s  risk={risk.item():.4f}  grad nonzero={int((Wsoft.grad.abs() > 0).sum())}")
print("OK")
