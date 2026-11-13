"""Gradient quality with TabPFN as the UQ module (analogue of Fig. 9 / Section 6.3).

At a fixed belief (the initial labelled data of a dataset) and uniform logits, we compare the
expected cosine similarity between a gradient estimate from N rollouts and a reference gradient
(REINFORCE with mean baseline and a very large number of rollouts), for the smoothed pathwise
estimator (tau in {0.1, 0.5, 2}) and REINFORCE with / without baseline, as a function of N.
All estimators use the Proposition-1 gain rollout value (squared change of the posterior mean of the
target, closed form from TabPFN's predictive moments).

  python gradient_quality.py --dataset eicu --gpu 0 --n_ref 20000 --reps 40
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.subset import gumbel_like, ordered_logprob, sample_hard_subset, soft_topk  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.uq_tabpfn import TabPFNModule  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, save_json, seed_all  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as reg_run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="eicu")
p.add_argument("--gpu", type=int, default=None)
p.add_argument("--n_ref", type=int, default=100000)
p.add_argument("--reps", type=int, default=40)
p.add_argument("--Ns", type=int, nargs="+", default=[1, 2, 4, 8, 10, 50, 100])
p.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.5, 2.0])
p.add_argument("--b", type=int, default=2)
p.add_argument("--pool_subset", type=int, default=50, help="restrict the pool to this many random items so that the reference is reliable")
p.add_argument("--batch", type=int, default=160, help="rollouts per TabPFN pass")
p.add_argument("--out", default=None)
args, _ = p.parse_known_args()
dev = pick_device(args.gpu)
torch.backends.cuda.matmul.allow_tf32 = True
seed_all(0)
ns = argparse.Namespace(dataset=args.dataset, seed_dataset=0, input_dim=10, n_train=100, n_pool=500, n_anchor=51, outputscale=0.69,
                        noise_std=0.1, center_scale=0.5, cluster_std=0.5, ls_per_sqrt_d=0.5, n_eval_max=285)
prob = reg_run.build_problem(ns, dev)
if args.pool_subset and args.pool_subset < prob.pool_x.shape[0]:
    keep = torch.randperm(prob.pool_x.shape[0], generator=torch.Generator().manual_seed(0))[: args.pool_subset].to(dev)
    prob.pool_x, prob.pool_y = prob.pool_x[keep], prob.pool_y[keep]
pred = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
target = MSETarget(predictor_outputs(pred, prob.eval_x, "regression"), 0.0)
uq = TabPFNModule("regression", device=dev, autocast=True, soft_topm=50)
st = uq.condition(prob.train_x, prob.train_y)
n = prob.pool_x.shape[0]
theta = torch.zeros(n, device=dev)
gen = torch.Generator(device=dev).manual_seed(1)
m_t = uq.target_mean(st, prob.eval_x, target).detach()
ybar = uq.sample_pool_labels(st, prob.pool_x, 1, gen)[0]  # imagined labels fixed (part of the fixed belief/rollout noise)


def gain_hard(idx_list):
    """rollout values (gain) for a list of hard batches; batched TabPFN passes."""
    out = []
    for s0 in range(0, len(idx_list), args.batch):
        W = torch.zeros(len(idx_list[s0: s0 + args.batch]), n, device=dev)
        for j, idx in enumerate(idx_list[s0: s0 + args.batch]):
            W[j, idx] = 1.0
        with torch.no_grad():
            mp = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, W, target, differentiable=False)
        out.append((mp - m_t) ** 2)
    return torch.cat(out)


def reinforce_grad(N, baseline, generator):
    th = theta.clone().requires_grad_(True)
    idxs = [sample_hard_subset(th.detach(), args.b, generator) for _ in range(N)]
    gains = gain_hard(idxs)  # maximise gain
    logps = torch.stack([ordered_logprob(th, idx) for idx in idxs])
    adv = gains - (gains.mean() if (baseline and N > 1) else 0.0)
    loss = -(adv * logps).mean()
    g, = torch.autograd.grad(loss, th)
    return -g  # ascent direction on the gain


def pathwise_grad(N, tau, generator, chunk=2):
    th = theta.clone().requires_grad_(True)
    noise = gumbel_like(th.detach().unsqueeze(0).expand(N, n), generator)
    total = torch.zeros(n, device=dev)
    for s0 in range(0, N, chunk):
        g_n = min(chunk, N - s0)
        A = soft_topk(th.unsqueeze(0).expand(g_n, n), args.b, tau, noise=noise[s0: s0 + g_n], clamp_max=1.0)
        mp = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, A, target, differentiable=True)
        gain = ((mp - m_t) ** 2).sum() / N
        g, = torch.autograd.grad(gain, th)
        total += g
    return total


print("computing reference gradient with", args.n_ref, "REINFORCE rollouts ...", flush=True)
ref = torch.zeros(n, device=dev)
chunks = 0
for s0 in range(0, args.n_ref, 4000):
    ref += reinforce_grad(min(4000, args.n_ref - s0), True, gen)
    chunks += 1
ref /= chunks
print("reference |grad| =", ref.norm().item(), " top-5 items:", torch.topk(ref, 5).indices.tolist(), flush=True)


def cos(a, b):
    na, nb = a.norm(), b.norm()
    return 0.0 if na == 0 or nb == 0 else float((a @ b) / (na * nb))


results = {"dataset": args.dataset, "n_ref": args.n_ref, "reps": args.reps, "Ns": args.Ns, "b": args.b, "pool": int(n), "cos": {}}
for N in args.Ns:
    for name, fn in ([("reinforce", lambda g: reinforce_grad(N, False, g)), ("reinforce_baseline", lambda g: reinforce_grad(N, True, g))]
                     + [(f"pathwise_tau{tau}", (lambda tau: lambda g: pathwise_grad(N, tau, g))(tau)) for tau in args.taus]):
        if name.startswith("pathwise") and N > 10:
            continue  # pathwise at N>10 is not needed for the comparison and is expensive
        vals = []
        for r in range(args.reps):
            g = torch.Generator(device=dev).manual_seed(1000 * N + r)
            vals.append(cos(fn(g), ref))
        results["cos"][f"{name}@N{N}"] = {"mean": sum(vals) / len(vals), "se": (torch.tensor(vals).std() / len(vals) ** 0.5).item()}
        print(f"N={N:4d} {name:22s} cos = {results['cos'][f'{name}@N{N}']['mean']:.3f} ± {results['cos'][f'{name}@N{N}']['se']:.3f}", flush=True)
out = args.out or f"/shared/share_mala/daksh/adaptive_sampling_experiments/results/tabpfn_regression/gradient_quality_{args.dataset}.json"
save_json(results, out)
print("saved", out)
