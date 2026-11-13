"""Multi-horizon acquisition loop shared by all experiments.

At each decision epoch t = 0..T-1 the belief is conditioned on the labelled data, the tracked
metrics are logged (posterior variance of the target = planning objective, posterior-mean
estimate of the target, and the realised target on the evaluation set), a batch of b pool items
is chosen by the method, the true labels are revealed and the items leave the pool.
"""
from __future__ import annotations

import time

import torch

from .planners import (PolicyGradientPlanner, select_random, select_uncertainty_sequential,
                       select_uncertainty_static)
from .targets import risk_and_estimate
from .utils import save_json

METHODS = ["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"]


def method_to_planner_kwargs(method: str):
    if method == "pathwise":
        return dict(estimator="pathwise")
    if method == "reinforce":
        return dict(estimator="reinforce", baseline="none")
    if method == "reinforce_baseline":
        return dict(estimator="reinforce", baseline="mean")
    if method == "reinforce_loo":
        return dict(estimator="reinforce", baseline="loo")
    return None


def track_metrics(uq, state, target, eval_x, eval_y, n_track, generator=None, truth=None):
    """truth: optional denoised value of the target (e.g. oracle_truth.oracle_target()['q_oracle'] for synthetic
    data); when given, the error against it is recorded next to the error against the realised labels."""
    with torch.no_grad():
        draws = uq.posterior_draws(state, eval_x, n_track, generator)
        Q = target.q(draws)
        var, est_mc = risk_and_estimate(Q)
        est = uq.target_mean(state, eval_x, target) if hasattr(uq, "target_mean") else est_mc
        actual = target.true_value(eval_y)
    out = {"var_target": float(var), "est_target": float(est), "est_target_mc": float(est_mc), "actual_target": actual,
           "abs_err": abs(float(est) - actual), "sq_err": (float(est) - actual) ** 2, "n_draws": int(Q.shape[-1])}
    if truth is not None:
        out.update({"actual_target_oracle": float(truth), "abs_err_oracle": abs(float(est) - float(truth)),
                    "sq_err_oracle": (float(est) - float(truth)) ** 2})
    return out


def run_acquisition(problem, uq, target, method: str, b: int, T: int, seed: int, out_path: str | None = None,
                    planner_kwargs: dict | None = None, n_track: int = 1000, verbose: bool = True, extra_record: dict | None = None,
                    truth: dict | None = None):
    """truth: optional dict from oracle_truth.oracle_target(); its 'q_oracle' is used as the denoised target value."""
    """Runs the T-epoch loop; returns the record dict (also saved to out_path if given)."""
    gen = torch.Generator().manual_seed(seed)
    dev = problem.train_x.device
    gdev = torch.Generator(device=dev).manual_seed(seed) if dev.type == "cuda" else gen
    train_x, train_y = problem.train_x.clone(), problem.train_y.clone()
    pool_x, pool_y = problem.pool_x.clone(), problem.pool_y.clone()
    pool_cluster = problem.pool_cluster.clone() if problem.pool_cluster is not None else None
    eval_x, eval_y = problem.eval_x, problem.eval_y
    record = {"method": method, "b": b, "T": T, "seed": seed, "problem": problem.summary(), "uq": uq.name,
              "planner_kwargs": planner_kwargs or {}, "horizon": [], "selected": [], "selected_clusters": [],
              "planner_history": [], "timing": []}
    if extra_record:
        record.update(extra_record)
    if truth is not None:
        record["truth"] = dict(truth)
    pk = method_to_planner_kwargs(method)
    t_run0 = time.time()
    for t in range(T + 1):
        t0 = time.time()
        state = uq.condition(train_x, train_y)
        # common random numbers for the tracked posterior draws: the same generator seed at every
        # epoch (and for every method with the same seed) so that curves are paired, not noisy
        gtrack = torch.Generator(device=dev).manual_seed(10_000 + seed) if dev.type == "cuda" else torch.Generator().manual_seed(10_000 + seed)
        m = track_metrics(uq, state, target, eval_x, eval_y, n_track, gtrack, truth=(truth or {}).get("q_oracle"))
        m["t"] = t
        m["n_labelled"] = int(train_x.shape[0])
        record["horizon"].append(m)
        if verbose:
            print(f"[{method} seed={seed}] t={t} n_lab={m['n_labelled']} var={m['var_target']:.4e} est={m['est_target']:.4f} "
                  f"actual={m['actual_target']:.4f} |err|={m['abs_err']:.4f}", flush=True)
        if t == T:
            break
        if method == "random":
            idx = select_random(pool_x.shape[0], b, gen)
        elif method == "uncertainty":
            idx = select_uncertainty_static(uq, state, pool_x, b)
        elif method == "uncertainty_seq":
            idx = select_uncertainty_sequential(uq, state, train_x, train_y, pool_x, b, gdev)
        elif pk is not None:
            kw = dict(pk)
            kw.update(planner_kwargs or {})
            planner = PolicyGradientPlanner(uq, target, b, **kw)
            idx, hist = planner.select(state, pool_x, eval_x, gdev)
            hist = {k: (v if k != "final_logits" else None) for k, v in hist.items()}
            record["planner_history"].append(hist)
        else:
            raise ValueError(method)
        idx = idx.cpu().long()
        record["selected"].append(idx.tolist())
        if pool_cluster is not None:
            record["selected_clusters"].append(pool_cluster[idx].cpu().tolist())
        train_x = torch.cat([train_x, pool_x[idx.to(dev)]], 0)
        train_y = torch.cat([train_y, pool_y[idx.to(dev)]], 0)
        keep = torch.ones(pool_x.shape[0], dtype=torch.bool)
        keep[idx] = False
        pool_x, pool_y = pool_x[keep.to(dev)], pool_y[keep.to(dev)]
        if pool_cluster is not None:
            pool_cluster = pool_cluster[keep.to(pool_cluster.device)]
        record["timing"].append(time.time() - t0)
        if out_path:
            save_json(record, out_path)
    record["total_time"] = time.time() - t_run0
    if out_path:
        save_json(record, out_path)
    return record
