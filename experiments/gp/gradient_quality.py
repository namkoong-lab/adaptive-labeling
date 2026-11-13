#!/usr/bin/env python
"""Gradient quality of the planners' estimators with the GP belief (the paper's cosine-similarity-vs-N figure).

At a fixed belief -- the initial state of a run.py problem (default: synthetic, d = 1, dataset seed 0), uniform logits over
the pool, one imagined label vector drawn from the posterior predictive -- every estimator targets the gradient of the
one-step objective the GP planners optimise (planners.PolicyGradientPlanner, objective 'risk'):

    H(theta) = E_{S ~ pi_theta} R(S),   R(S) = posterior variance of the target Q after labelling S.

With R_hat the planners' Monte Carlo estimate of R (sample variance of Q over --n_mc posterior draws):
    reinforce            (1/N) sum_i R_hat(S_i) grad log pi(S_i)
    reinforce_baseline   (1/N) sum_i (R_hat(S_i) - mean_j R_hat(S_j)) grad log pi(S_i)        (zero at N = 1)
    pathwise_tau<t>      (1/N) sum_i grad R_hat(soft_topk(theta, b, t; Gumbel noise i))        (biased: smoothed objective)
The true gradient is approximated by REINFORCE with a mean baseline over --n_ref rollouts, using the closed form of R for
the MSE target, Var Q = (2 ||Sigma||_F^2 + 4 a' Sigma a) / N_eval^2 with a = mu - psi, which is the expectation of R_hat
(so the reference carries no Monte Carlo noise from the posterior draws).

Reported: the expected cosine similarity between an N-rollout estimate and the true gradient (0 for a zero estimate), over
--reps estimates per N.  Each estimator's per-rollout terms are drawn once into a bank of --bank independent rollouts; an
N-rollout estimate uses N distinct bank entries, so it has exactly the distribution of a fresh estimate (estimates for
large N share some entries, which only makes their standard error optimistic).

  python gradient_quality.py --gpu 0
  python -m adaptive_sampling_lib.plot_gradient_quality --json <output json>
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root
from adaptive_sampling_lib.paths import results  # noqa: E402
from adaptive_sampling_lib.subset import gumbel_like, ordered_logprob, soft_topk  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, save_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("gp_run", os.path.join(HERE, "run.py"))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
JITTER = 1e-5  # GPModule._draw adds this to the posterior covariance before drawing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "eicu"])
    p.add_argument("--input_dim", type=int, default=1)
    p.add_argument("--seed_dataset", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="imagined labels, rollouts and Monte Carlo draws")
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_mc", type=int, default=100, help="posterior draws per rollout (as the planners)")
    p.add_argument("--Ns", type=int, nargs="+", default=[1, 2, 4, 8, 10, 100, 1000])
    p.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--reps", type=int, default=200, help="estimates per N")
    p.add_argument("--bank", type=int, default=20000, help="independent rollouts per estimator (>= max N)")
    p.add_argument("--n_ref", type=int, default=1000000, help="rollouts of the reference gradient")
    p.add_argument("--chunk", type=int, default=2000, help="hard rollouts per batch")
    p.add_argument("--pw_chunk", type=int, default=100, help="pathwise rollouts per batch")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    assert a.bank >= max(a.Ns)
    dev = pick_device(a.gpu)
    rargs = run.add_args(argparse.ArgumentParser()).parse_args(
        ["--method", "pathwise", "--dataset", a.dataset, "--input_dim", str(a.input_dim), "--seed_dataset", str(a.seed_dataset)])
    prob, target, uq, _ = run.build(rargs, dev)
    state = uq.condition(prob.train_x, prob.train_y)
    pool_x, eval_x = prob.pool_x, uq._t(prob.eval_x)
    n, b = pool_x.shape[0], a.b
    psi = target.pred.to(eval_x)
    theta = torch.full((n,), math.log(1.0 / n), device=dev)  # the planners' initial (uniform) logits
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    ybar = uq.sample_pool_labels(state, pool_x, 1, gen)[0]  # one imagined label vector, shared by every rollout
    I_eval = torch.eye(eval_x.shape[0], device=dev, dtype=eval_x.dtype)

    def hard_subsets(B, g):  # B ordered draws from pi_theta (Gumbel-top-b, as subset.sample_hard_subset)
        th = theta.expand(B, n)
        return torch.topk(th + gumbel_like(th, g), b).indices

    def onehot(idx):
        return torch.zeros(idx.shape[0], n, device=dev).scatter_(1, idx, 1.0)

    def score(idx):  # grad_theta log pi^ord(idx) for idx [B, b] (vectorised subset.ordered_logprob)
        th = theta.double().expand(idx.shape[0], n)
        g = torch.zeros_like(th)
        mask = torch.zeros_like(th, dtype=torch.bool)
        for j in range(b):
            g -= torch.softmax(th.masked_fill(mask, float("-inf")), -1)
            g.scatter_add_(1, idx[:, j: j + 1], torch.ones_like(g[:, :1]))
            mask.scatter_(1, idx[:, j: j + 1], True)
        return g

    def risk_exact(idx):  # closed-form R(S) for hard subsets idx [B, b]
        Xg, yg, wg = uq._rollout_inputs(state, pool_x, ybar, onehot(idx), differentiable=False)
        mu, cov = uq.weighted_posterior(Xg, yg, wg, eval_x)
        cov = cov + JITTER * I_eval
        r = mu - psi
        return (2 * (cov ** 2).sum((-1, -2)) + 4 * torch.einsum("bi,bij,bj->b", r, cov, r)) / eval_x.shape[0] ** 2

    def risk_mc(W, differentiable, g):  # the planners' R_hat for weights W [B, n]
        draws = uq.rollout_draws(state, eval_x, pool_x, ybar, W, a.n_mc, differentiable=differentiable, generator=g)
        return target.q(draws).var(-1)

    # ---- checks: the vectorised score and the closed form against the library's ordered_logprob and the MC risk
    idx = hard_subsets(4, gen)
    th = theta.clone().requires_grad_(True)
    auto = torch.stack([torch.autograd.grad(ordered_logprob(th, i), th)[0] for i in idx]).double()
    err_score = (auto - score(idx)).abs().max().item()
    with torch.no_grad():
        mc = torch.stack([risk_mc(onehot(idx), False, gen) for _ in range(400)]).mean(0)
        ex = risk_exact(idx)
    print(f"[check] score vs autograd: max |diff| {err_score:.1e}; closed-form risk / mean of 400 MC risks: "
          f"{(ex / mc).cpu().numpy().round(3)} (MC se ~{math.sqrt(2 / (a.n_mc - 1) / 400):.3f})", flush=True)
    assert err_score < 1e-6

    # ---- reference gradient: REINFORCE with a mean baseline, closed-form R, n_ref rollouts
    t0 = time.time()
    gref = torch.Generator(device=dev).manual_seed(a.seed + 1)
    idxs, Rs = [], []
    with torch.no_grad():
        for s in range(0, a.n_ref, a.chunk):
            i = hard_subsets(min(a.chunk, a.n_ref - s), gref)
            idxs.append(i)
            Rs.append(risk_exact(i))
    R = torch.cat(Rs)
    Rbar = R.mean()
    acc = torch.zeros(n, dtype=torch.float64, device=dev)
    sq = 0.0
    for i, r in zip(idxs, Rs):
        x = (r - Rbar).unsqueeze(-1) * score(i)
        acc += x.sum(0)
        sq += float((x ** 2).sum())
    g_true = acc / a.n_ref / (1 - 1 / a.n_ref)  # E[(R_i - Rbar) s_i] = (1 - 1/M) grad H
    se = math.sqrt(max(sq / a.n_ref - float((acc / a.n_ref).pow(2).sum()), 0.0) / a.n_ref)
    print(f"[ref] {a.n_ref} rollouts in {time.time() - t0:.0f}s: |grad H| = {g_true.norm():.3e}, noise/signal = {se / g_true.norm():.3f}"
          f"  (E R = {Rbar:.4e})", flush=True)
    del idxs, Rs, R

    # ---- banks of per-rollout terms
    t0 = time.time()
    hard_R, hard_S = [], []
    with torch.no_grad():
        for s in range(0, a.bank, a.chunk):
            i = hard_subsets(min(a.chunk, a.bank - s), gen)
            hard_R.append(risk_mc(onehot(i), False, gen).double())
            hard_S.append(score(i))
    hard_R, hard_S = torch.cat(hard_R), torch.cat(hard_S)
    print(f"[bank] REINFORCE: {a.bank} rollouts in {time.time() - t0:.0f}s", flush=True)
    pw = {}
    for tau in a.taus:
        t0 = time.time()
        grads = []
        for s in range(0, a.bank, a.pw_chunk):
            B = min(a.pw_chunk, a.bank - s)
            th = theta.detach().expand(B, n).clone().requires_grad_(True)  # one row per rollout -> per-rollout gradients
            A = soft_topk(th, b, tau, noise=gumbel_like(th.detach(), gen), clamp_max=1.0)
            risk_mc(A, True, gen).sum().backward()
            grads.append(th.grad.double())
        pw[tau] = torch.cat(grads)
        print(f"[bank] pathwise tau={tau}: {a.bank} rollouts in {time.time() - t0:.0f}s", flush=True)

    # ---- N-rollout estimates
    def cos(v):
        nv = v.norm()
        return 0.0 if nv == 0 else float(v @ g_true / (nv * g_true.norm()))

    names = ["reinforce", "reinforce_baseline"] + [f"pathwise_tau{t}" for t in a.taus]
    res = {"belief": "GP", "dataset": a.dataset if a.dataset == "eicu" else f"synthetic, d = {a.input_dim}",
           "seed_dataset": a.seed_dataset, "n_ref": a.n_ref, "reps": a.reps, "bank": a.bank, "Ns": a.Ns, "b": b, "pool": n,
           "n_mc": a.n_mc, "taus": a.taus, "ref_noise_to_signal": se / float(g_true.norm()), "cos": {}, "cos_of_bank_mean": {}}
    bank_mean = {"reinforce": (hard_R.unsqueeze(-1) * hard_S).mean(0),
                 "reinforce_baseline": ((hard_R - hard_R.mean()).unsqueeze(-1) * hard_S).mean(0)}
    bank_mean.update({f"pathwise_tau{t}": pw[t].mean(0) for t in a.taus})
    for name in names:  # unbiased estimators: ~1 up to bank noise; pathwise: the smoothing bias
        res["cos_of_bank_mean"][name] = cos(bank_mean[name])
    gs = torch.Generator(device=dev).manual_seed(a.seed + 2)
    for N in a.Ns:
        vals = {k: [] for k in names}
        for _ in range(a.reps):
            sel = torch.randperm(a.bank, generator=gs, device=dev)[:N]
            Rn, Sn = hard_R[sel], hard_S[sel]
            vals["reinforce"].append(cos((Rn.unsqueeze(-1) * Sn).mean(0)))
            vals["reinforce_baseline"].append(cos(((Rn - Rn.mean()).unsqueeze(-1) * Sn).mean(0)))
            for t in a.taus:
                vals[f"pathwise_tau{t}"].append(cos(pw[t][sel].mean(0)))
        for k in names:
            v = torch.tensor(vals[k])
            res["cos"][f"{k}@N{N}"] = {"mean": float(v.mean()), "se": float(v.std() / math.sqrt(len(v)))}
        print(f"N={N:5d}  " + "  ".join(f"{k} {res['cos'][f'{k}@N{N}']['mean']:+.3f}" for k in names), flush=True)
    print("cos(bank mean, true gradient): " + "  ".join(f"{k} {v:+.3f}" for k, v in res["cos_of_bank_mean"].items()))
    tag = "eicu" if a.dataset == "eicu" else f"dim{a.input_dim}_ds{a.seed_dataset}"
    out = a.out or results("gp_high_dim", "gradient_quality", f"gradient_quality_{tag}.json")
    save_json(res, out)
    print("saved", out)


if __name__ == "__main__":
    main()
