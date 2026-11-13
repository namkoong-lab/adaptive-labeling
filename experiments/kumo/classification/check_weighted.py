#!/usr/bin/env python
"""Checks of the weighted Kumo-Tabular classifier belief (adaptive_sampling_lib.uq_kumo, task='classification', the
labelled data's preprocessing statistics in the pseudo-updates) on one final-grid split:
  1. exactness: the unweighted module against the package's own forward with the same single-member recipe
  2. hard = soft: binary weights in a 50-item soft context against the hard update with the 5 selected items
  3. continuity: all pool weights ~0 against M_t, and one item at weight 1e-6 against that item removed
  4. gradient: autograd derivative of E[Q] along a random direction in the weights against a central finite difference
  5. precision: class probabilities under float16 autocast (the draws' setting) against float32
  6. cost: one pathwise step chunk (2 rollouts, forward + backward) and 128 tracked joint draws
  python check_weighted.py --gpu 0 [--dataset magic] [--size medium]"""
import argparse, importlib.util, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))  # repo root
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.subset import soft_topk  # noqa: E402
from adaptive_sampling_lib.targets import LogLossTarget  # noqa: E402
spec = importlib.util.spec_from_file_location("kumo_cls_run", os.path.join(HERE, "run.py")); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=0); p.add_argument("--dataset", default="magic"); p.add_argument("--size", default="medium")
a = p.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
q = argparse.ArgumentParser(); run.add_args(q)
args = q.parse_args(["--dataset", a.dataset, "--layout", "skewed", "--method", "pathwise", "--seed_dataset", "1", "--kumo_size", a.size])
torch.manual_seed(1)
prob = run.base.build_problem(args, dev)
uq = run.build_uq(args, prob, dev)
target = LogLossTarget(predictor_outputs(train_predictor(prob.train_x, prob.train_y, "classification", seed=42, device=dev, n_classes=2), prob.eval_x, "classification"))
st = uq.condition(prob.train_x, prob.train_y)
ok = True

# 1. exactness against the package
import sdm.processing as sp  # noqa: E402
from sdm.models import KumoTabular  # noqa: E402
num = sp.Sequential(sp.Cast(torch.float64), sp.DropConstantColumns(), sp.Standardize(eps=1e-6), sp.Clip(-100.0, 100.0), sp.ClipSigma(threshold=4.0))
recipe = sp.Recipe(features=[sp.StypeDispatch(numerical=[num]), sp.Cast(torch.float32)], target=[sp.StypeDispatch(categorical=[sp.AlignCategories(sort_by="value")])],
                   output=sp.TaskDispatch(classification=[sp.AverageEstimators(), sp.Softmax()]))
wrapper = KumoTabular(task="classification", size=a.size, device=dev); wrapper.eval()
with torch.no_grad():
    ref = torch.tensor(wrapper(prob.train_x.float(), prob.train_y.long().unsqueeze(-1), prob.eval_x.float(), recipe=recipe, num_estimators=1).to_pandas().to_numpy(), device=dev)
ours = uq.predictive(st, prob.eval_x)
d1 = (ours - ref).abs().max().item()
print(f"1. package vs module: max |diff| of class probabilities {d1:.2e}"); ok &= d1 < 1e-4

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

# 3. continuity
m_t = uq.target_mean(st, prob.eval_x, target).item()
z = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, torch.full((1, n), 1e-12, device=dev), target, differentiable=True).item()
W3 = W[:1].clone(); j = int(W3[0].nonzero()[0]); W3[0, j] = 1e-6
s3 = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, W3, target, differentiable=True).item()
keep = W[0].nonzero().squeeze(1); keep = keep[keep != j]
rem = uq._target_mean(uq.forward_raw(torch.cat([st["X"], prob.pool_x[keep]]), torch.cat([st["y"], ybar[keep].float()]), prob.eval_x, st["keep"], stats=st["stats"]), target)[0].item()
print(f"3. all weights ~0: {z:.6f} vs M_t {m_t:.6f} (|diff| {abs(z - m_t):.1e}); weight 1e-6: {s3:.6f} vs item removed {rem:.6f} (|diff| {abs(s3 - rem):.1e}; full item {hard[0].item():.6f})")
ok &= abs(z - m_t) < 1e-5 and abs(s3 - rem) < 1e-5

# 4. gradient against a central finite difference, along directions that move only items strictly inside the 50-item soft
#    context (items near the 51st weight would leave or enter it: soft_topm truncation, a jump the gradient does not see)
theta = torch.randn(n, device=dev, generator=gen)
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

# 5. float16 autocast (draws) against float32
with torch.autocast("cuda", dtype=torch.float16):
    p16 = uq.predictive(st, prob.eval_x)
d5 = (p16 - ours).abs()
print(f"5. float16 vs float32 class probabilities: max |diff| {d5.max().item():.1e}, mean {d5.mean().item():.1e}; E[Q] {target.mean_from_probs(p16).item():.5f} vs {target.mean_from_probs(ours).item():.5f}")

# 6. cost (TF32 as in the runs)
torch.backends.cuda.matmul.allow_tf32 = True
X6 = torch.cat([prob.train_x, prob.pool_x[:45]]); y6 = torch.cat([prob.train_y.float(), ybar[:45].float()]); st6 = uq.condition(X6, y6)
A2 = soft_topk(theta.unsqueeze(0).expand(2, n), 5, 0.5, noise=torch.zeros(2, n, device=dev), clamp_max=1.0).detach().requires_grad_(True)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
for rep in range(3):
    t0 = time.time(); uq.rollout_target_mean(st6, prob.eval_x, prob.pool_x, ybar, A2, target, differentiable=True).sum().backward(); torch.cuda.synchronize(); dt = time.time() - t0
print(f"6. pathwise step chunk (2 rollouts, context {X6.shape[0]} + 50 soft items): {dt:.2f}s, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
t0 = time.time(); dr = uq.posterior_draws(st6, prob.eval_x, 128, gen); torch.cuda.synchronize()
print(f"   128 tracked joint draws: {time.time() - t0:.2f}s, draws {tuple(dr.shape)}, Q over draws: mean {target.q(dr).mean().item():.4f} sd {target.q(dr).std().item():.4f}")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
