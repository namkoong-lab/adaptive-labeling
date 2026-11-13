#!/usr/bin/env python
"""Checks of the weighted Kumo-Tabular belief (adaptive_sampling_lib.uq_kumo) on one final-grid split:
  1. exactness: the unweighted module against the package's own forward with the same single-member recipe
     (identity transform, no random column shuffle or sign flips)
  2. hard = soft: binary weights in a 50-item soft context against the hard update with the 5 selected items
  3. continuity: an item with weight 1e-6 against the item removed
  4. gradient: autograd derivative of E[Q] along a random direction in the weights against a central finite difference
  5. cost: time and peak memory of one pathwise step (2 rollouts, forward + backward) and of 128 tracked joint draws
  python check_weighted.py --gpu 0 [--dataset eicu_raw] [--size medium]"""
import argparse, importlib.util, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))  # repo root
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.subset import soft_topk  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
spec = importlib.util.spec_from_file_location("kumo_run", os.path.join(HERE, "run.py")); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=0); p.add_argument("--dataset", default="eicu_raw"); p.add_argument("--size", default="medium")
a = p.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
q = argparse.ArgumentParser(); run.add_args(q)
args = q.parse_args(["--dataset", a.dataset, "--layout", "skewed", "--moments", "range", "--method", "pathwise", "--seed_dataset", "1", "--kumo_size", a.size])
torch.manual_seed(1)
prob = run.base.build_problem(args, dev)
uq = run.build_uq(args, prob, dev)
psi = predictor_outputs(train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev), prob.eval_x, "regression")
target = MSETarget(psi, noise_var=0.0)
st = uq.condition(prob.train_x, prob.train_y)
ok = True

# 1. exactness against the package
import sdm.processing as sp  # noqa: E402
from sdm.models import KumoTabular  # noqa: E402
num = sp.Sequential(sp.Cast(torch.float64), sp.DropConstantColumns(), sp.Standardize(eps=1e-6), sp.Clip(-100.0, 100.0), sp.ClipSigma(threshold=4.0))
recipe = sp.Recipe(features=[sp.StypeDispatch(numerical=[num]), sp.Cast(torch.float32)], target=[sp.StypeDispatch(numerical=[sp.Standardize()])],
                   output=sp.TaskDispatch(regression=[sp.SortQuantiles(), sp.AverageEstimators(trim_fraction=0.2)]))
wrapper = KumoTabular(task="regression", size=a.size, device=dev); wrapper.eval()
with torch.no_grad():
    ref = torch.tensor(wrapper(prob.train_x.float(), prob.train_y.float().unsqueeze(-1), prob.eval_x.float(), recipe=recipe, num_estimators=1).to_pandas().to_numpy(), device=dev)
ours = uq.forward_raw(st["X"], st["y"], prob.eval_x, st["keep"])[0].float()
d1 = (ours - ref).abs().max().item(); sc = (ref[:, -1] - ref[:, 0]).abs().mean().item()
print(f"1. package vs module: max |diff| {d1:.2e} (mean 0.1-99.9% width {sc:.3f})"); ok &= d1 < 1e-3 * sc

# 2. hard = soft with binary weights
gen = torch.Generator(device=dev).manual_seed(0)
ybar = uq.sample_pool_labels(st, prob.pool_x, 1, gen)[0]
n = prob.pool_x.shape[0]; G = 2
W = torch.zeros(G, n, device=dev)
for g in range(G):
    W[g, torch.randperm(n, generator=torch.Generator().manual_seed(g))[:5].to(dev)] = 1.0
hard = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, W, target, differentiable=False)
soft = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, W, target, differentiable=True).detach()
d2 = (hard - soft).abs().max().item()
print(f"2. hard {hard.tolist()} vs soft {soft.tolist()}: max |diff| {d2:.2e}"); ok &= d2 < 1e-4 * hard.abs().max().item()

# 3. continuity: one selected item down-weighted to 1e-6 against that item removed
W3 = W.clone(); j = int(W3[0].nonzero()[0]); W3[0, j] = 1e-6
soft3 = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, W3[:1], target, differentiable=True).detach()
keep = W[0].nonzero().squeeze(1); keep = keep[keep != j]
X4 = torch.cat([st["X"], prob.pool_x[keep]]); y4 = torch.cat([st["y"], ybar[keep]])
st4 = uq.condition(X4, y4); st4["keep"] = st["keep"]
rem = uq._target_mean(uq.forward_raw(st4["X"], st4["y"], prob.eval_x, st4["keep"]), target)[0]
d3 = abs(soft3.item() - rem.item())
print(f"3. weight 1e-6: {soft3.item():.6f} vs removed {rem.item():.6f}: |diff| {d3:.2e} (vs full item {hard[0].item():.6f})"); ok &= d3 < 1e-3 * abs(rem.item())

# 4. gradient (full moments; range moments are piecewise constant in the count of quantiles inside the range) against a
#    central finite difference, along directions that move only items strictly inside the 50-item soft
#    context (items near the 51st weight would leave or enter it: soft_topm truncation, a jump the gradient does not see)
theta = torch.randn(n, device=dev, generator=gen)
yr = uq.y_range; uq.y_range = None
A = soft_topk(theta.unsqueeze(0), 5, 0.5, noise=torch.zeros(1, n, device=dev), clamp_max=1.0).detach()
w = A[0]; top = torch.topk(w, 50).indices; w51 = torch.topk(w, 51).values[-1].item()
inner = top[(w[top] > w51 + 0.02) & (w[top] < 0.98)]
for rep in range(3):
    dirn = torch.zeros(n, device=dev); dirn[inner] = torch.randn(inner.numel(), device=dev, generator=gen); dirn /= dirn.norm()
    Aw = A.clone().requires_grad_(True)
    f = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, Aw, target, differentiable=True)[0]; f.backward()
    gd = float((Aw.grad[0] * dirn).sum())
    fds = []
    for eps in (1e-2, 1e-3):
        fp = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, A + eps * dirn, target, differentiable=True)[0].item()
        fm = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, A - eps * dirn, target, differentiable=True)[0].item()
        fds.append((fp - fm) / (2 * eps))
    rel = abs(gd - fds[1]) / max(abs(fds[1]), 1e-12)
    print(f"4. direction {rep} ({inner.numel()} interior items): autograd {gd:+.4e} vs finite difference {fds[0]:+.4e} (eps 1e-2), {fds[1]:+.4e} (eps 1e-3): rel. diff {rel:.2%}")
    ok &= rel < 0.05

uq.y_range = yr

# 5. cost (TF32 as in the runs)
if os.environ.get('SKIP_COST'):
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'); sys.exit()
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
X5 = torch.cat([prob.train_x, prob.pool_x[:45]]); y5 = torch.cat([prob.train_y, ybar[:45]]); st5 = uq.condition(X5, y5)  # a late-epoch context (t = 9)
A2 = soft_topk(theta.unsqueeze(0).expand(2, n), 5, 0.5, noise=torch.zeros(2, n, device=dev), clamp_max=1.0).detach().requires_grad_(True)
for rep in range(3):
    t0 = time.time(); fq = uq.rollout_target_mean(st5, prob.eval_x, prob.pool_x, ybar, A2, target, differentiable=True); fq.sum().backward(); torch.cuda.synchronize(); dt = time.time() - t0
print(f"5. pathwise step chunk (2 rollouts, context {X5.shape[0]} + 50 soft items, 285 eval points): {dt:.2f}s, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time(); dr = uq.posterior_draws(st5, prob.eval_x, 128, gen); torch.cuda.synchronize()
print(f"   128 tracked joint draws (4 chunks): {time.time() - t0:.2f}s, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB, draws {tuple(dr.shape)}")
t0 = time.time(); pl = uq.sample_pool_labels(st5, prob.pool_x, 1, gen); torch.cuda.synchronize()
print(f"   imagined pool labels (5 chunks): {time.time() - t0:.2f}s")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
