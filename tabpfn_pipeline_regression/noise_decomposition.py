#!/usr/bin/env python
"""Decompose the noise of the pathwise planner's per-step objective on TabPFN (eICU raw, skewed layout) into its two
sources: the imagined pool labels (one joint fantasy per step) and the Gumbel noise of the G soft subsets.
For a fixed logit vector theta the loss is evaluated repeatedly with (a) fixed fantasy / fresh Gumbel, (b) fresh fantasy /
fixed Gumbel, (c) both fresh (the planner's situation), (d) both fresh with one fantasy per rollout; the gradient is also
re-estimated under (c) and (d) to measure how reproducible the one-step pick (top-b by gradient) is.
Also checks that per-rollout labels with identical rows reproduce the shared-fantasy loss exactly."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from run import build_problem, build_uq, add_args  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.planners import PolicyGradientPlanner  # noqa: E402
from adaptive_sampling_lib.subset import soft_topk, gumbel_like  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

p = argparse.ArgumentParser(); add_args(p)
p.add_argument("--reps", type=int, default=12)
p.add_argument("--ref_json", default="/shared/share_mala/daksh/adaptive_sampling_experiments/results/tabpfn_regression/skew_splits_low/tabpfn_eicu_raw_skewed/pathwise_ds3_s3.json")
p.add_argument("--out", default="/shared/share_mala/daksh/adaptive_sampling_experiments/results/tabpfn_regression/noise_decomposition_eicu.json")
a = p.parse_args(["--dataset", "eicu_raw", "--layout", "skewed", "--method", "pathwise", "--seed_dataset", "3", "--seed", "3"] + sys.argv[1:])
dev = pick_device(a.gpu); seed_all(a.seed)
prob = build_problem(a, dev)
predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
psi = predictor_outputs(predictor, prob.eval_x, "regression")
target = MSETarget(psi, noise_var=0.0)
uq = build_uq(a, prob, dev)
ref = json.load(open(a.ref_json))
planner = PolicyGradientPlanner(uq, target, a.b, n_steps=1, G=a.G, n_mc=a.n_mc, lr=a.lr, tau=a.tau, estimator="pathwise", objective="gain", grad_chunk=a.grad_chunk)
G, b, tau = planner.G, planner.b, planner.tau


def loss_and_grad(theta, state, ybar, noise, grad=True):
    th = theta.detach().clone().requires_grad_(grad)
    n = th.shape[0]; risks = []
    gc = planner.grad_chunk or G
    for s0 in range(0, G, gc):
        g_n = min(gc, G - s0)
        A = soft_topk(th.unsqueeze(0).expand(g_n, n), b, tau, noise=noise[s0: s0 + g_n], clamp_max=1.0)
        yb = ybar[s0: s0 + g_n] if ybar.dim() == 2 else ybar
        r, _ = planner._risk(state, prob.eval_x, pool_x, yb, A, differentiable=grad)
        if grad: (r.sum() / G).backward()
        risks.append(r.detach())
    return float(torch.cat(risks).mean()), (th.grad.detach().clone() if grad else None)


def stats(vals):
    v = np.array(vals); return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), rel=float(v.std(ddof=1) / abs(v.mean())))


out = {"args": vars(a), "epochs": {}}
train_x, train_y, pool_x = prob.train_x.clone(), prob.train_y.clone(), prob.pool_x.clone()
pool_mask = torch.ones(pool_x.shape[0], dtype=torch.bool)
gen = torch.Generator(device=dev).manual_seed(123)
for epoch in range(4):
    if epoch in (0, 3):
        t0 = time.time()
        state = uq.condition(train_x, train_y)
        with torch.no_grad():
            planner._m_t = uq.target_mean(state, prob.eval_x, target).detach()
        n = pool_x.shape[0]
        thetas = {"uniform": torch.full((n,), float(np.log(1 / n)), device=dev)}
        pick = torch.tensor(ref["selected"][epoch], device=dev)  # the stored one-step pick at this epoch (pool indices at that epoch)
        peaked = thetas["uniform"].clone(); peaked[pick] += 3.0; thetas["peaked"] = peaked
        res = {}
        for tname, theta in thetas.items():
            yb0 = uq.sample_pool_labels(state, pool_x, 1, gen)[0]; nz0 = gumbel_like(theta.unsqueeze(0).expand(G, n), gen)
            # plumbing check: per-rollout labels with identical rows == shared labels
            l_sh, g_sh = loss_and_grad(theta, state, yb0, nz0); l_pr, g_pr = loss_and_grad(theta, state, yb0.unsqueeze(0).expand(G, -1).contiguous(), nz0)
            res[tname] = {"check_identical_rows": {"loss_shared": l_sh, "loss_per_rollout": l_pr, "grad_maxabsdiff": float((g_sh - g_pr).abs().max()), "grad_maxabs": float(g_sh.abs().max())}}
            conds = {"fixed_fantasy_fresh_gumbel": lambda: (yb0, gumbel_like(theta.unsqueeze(0).expand(G, n), gen)),
                     "fresh_fantasy_fixed_gumbel": lambda: (uq.sample_pool_labels(state, pool_x, 1, gen)[0], nz0),
                     "both_fresh_shared": lambda: (uq.sample_pool_labels(state, pool_x, 1, gen)[0], gumbel_like(theta.unsqueeze(0).expand(G, n), gen)),
                     "both_fresh_per_rollout": lambda: (uq.sample_pool_labels(state, pool_x, G, gen), gumbel_like(theta.unsqueeze(0).expand(G, n), gen))}
            for cname, draw in conds.items():
                losses, grads = [], []
                for _ in range(a.reps):
                    yb, nz = draw(); l, g = loss_and_grad(theta, state, yb, nz); losses.append(l); grads.append(g)
                Gm = torch.stack(grads); cos = []; overlap = []
                for i in range(len(grads)):
                    for j in range(i + 1, len(grads)):
                        cos.append(float(torch.nn.functional.cosine_similarity(Gm[i], Gm[j], dim=0)))
                        ti, tj = set(torch.topk(-Gm[i], b).indices.tolist()), set(torch.topk(-Gm[j], b).indices.tolist()); overlap.append(len(ti & tj) / b)
                mean_grad = Gm.mean(0); cos_mean = [float(torch.nn.functional.cosine_similarity(g, mean_grad, dim=0)) for g in grads]
                res[tname][cname] = {"loss": stats(losses), "grad_pairwise_cos": float(np.mean(cos)), "grad_cos_to_mean": float(np.mean(cos_mean)), "pick_overlap": float(np.mean(overlap))}
                print(f"epoch {epoch} theta={tname:8s} {cname:28s} loss {res[tname][cname]['loss']['mean']:+.4e} rel-std {res[tname][cname]['loss']['rel']:.2f}  grad cos {np.mean(cos):.2f}  pick overlap {np.mean(overlap):.2f}", flush=True)
        out["epochs"][str(epoch)] = res; print(f"epoch {epoch} done in {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    # advance the state along the stored one-step trajectory
    idx = torch.tensor(ref["selected"][epoch], device=dev)
    train_x = torch.cat([train_x, pool_x[idx]], 0); train_y = torch.cat([train_y, prob.pool_y[pool_mask.to(dev)][idx]], 0)
    keep = torch.ones(pool_x.shape[0], dtype=torch.bool, device=dev); keep[idx] = False
    pool_x = pool_x[keep]
    pm_idx = pool_mask.nonzero().squeeze(1)[idx.cpu()]; pool_mask[pm_idx] = False
print("saved", a.out)
