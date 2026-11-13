#!/usr/bin/env python
"""Checks of the Kumo-Tabular relaxations (adaptive_sampling_lib.uq_kumo: relax weighted / weighted with full self-attention
/ avg_input / avg_emb) for one task on one final-grid split (t=0):
  1. exactness: binary weights (weighted) or one-hot slots (avg_*) against the hard update, E[Q] for two random batches
  2. gradient: autograd derivative of E[Q] along a random direction in the planner's logits against a central finite
     difference (fixed Gumbel noise; directions on the items inside the soft context)
  3. the risk objective: posterior variance of Q from rollout_draws, hard and differentiable at binary weights (same seed),
     and a finite gradient through the differentiable draws
  4. cost: seconds per planner step (10 rollouts in chunks of 2, forward + backward) for each objective
  python check_relaxations.py --task regression|classification --gpu 0 [--dataset california] [--split 1]"""
import argparse, importlib.util, math, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.subset import soft_topk, soft_topk_slots, gumbel_like  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget, LogLossTarget, risk_and_estimate  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--task", default="regression"); p.add_argument("--gpu", type=int, default=0)
p.add_argument("--dataset", default=None); p.add_argument("--split", type=int, default=1); p.add_argument("--skip_risk", action="store_true"); p.add_argument("--skip_cost", action="store_true")
a = p.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
CLS = a.task == "classification"
ds = a.dataset or ("mimic" if CLS else "california")
path = os.path.join(HERE, "..", "kumo_pipeline_classification/run.py" if CLS else "kumo_pipeline_regression/run.py")
spec = importlib.util.spec_from_file_location("runmod", path); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
VARIANTS = [("weighted", "weighted"), ("weighted", "full"), ("avg_input", "weighted"), ("avg_emb", "weighted")]
ok = True
for relax, sa in VARIANTS:
    torch.backends.cuda.matmul.allow_tf32 = False
    q = argparse.ArgumentParser(); run.add_args(q)
    argv = ["--dataset", ds, "--layout", "skewed", "--method", "pathwise", "--seed_dataset", str(a.split), "--kumo_relax", relax, "--kumo_self_attn", sa, "--kumo_fixed_stats", "both"]
    args = q.parse_args(argv + ([] if CLS else ["--moments", "range"]))
    torch.manual_seed(a.split); prob = run.base.build_problem(args, dev); uq = run.build_uq(args, prob, dev)
    kind = "classification" if CLS else "regression"
    out = predictor_outputs(train_predictor(prob.train_x, prob.train_y, kind, seed=42, device=dev, **({"n_classes": 2} if CLS else {})), prob.eval_x, kind)
    target = LogLossTarget(out) if CLS else MSETarget(out, noise_var=0.0)
    st = uq.condition(prob.train_x, prob.train_y); n = prob.pool_x.shape[0]
    gen = torch.Generator(device=dev).manual_seed(0)
    ybar = uq.sample_pool_labels(st, prob.pool_x, 1, gen)[0].float()
    name = f"{relax}" + (" + full self-attention" if sa == "full" else "")
    batches = [torch.randperm(n, generator=torch.Generator().manual_seed(g))[:5].to(dev) for g in range(2)]
    Wb = torch.zeros(2, n, device=dev)
    for g, b in enumerate(batches):
        Wb[g, b] = 1.0
    with torch.no_grad():
        hard = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, Wb, target, differentiable=False)
    if uq.slot_relaxation:
        Wr = torch.zeros(2, 5, n, device=dev)
        for g, b in enumerate(batches):
            Wr[g, torch.arange(5), b] = 1.0
    else:
        Wr = Wb
    soft = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, Wr, target, differentiable=True).detach()
    d1 = (hard - soft).abs().max().item()
    # 2. gradient along logits (regression: full moments; range-conditioned moments are piecewise constant in the number of
    #    quantiles inside the range, so their finite differences jump)
    y_range, uq.y_range = uq.y_range, None
    theta = torch.randn(n, device=dev, generator=gen) * 0.5; noise = gumbel_like(theta.unsqueeze(0), gen)
    relaxed = (lambda th: soft_topk_slots(th.unsqueeze(0), 5, 0.5, noise=noise)) if uq.slot_relaxation else (lambda th: soft_topk(th.unsqueeze(0), 5, 0.5, noise=noise, clamp_max=1.0))
    w = relaxed(theta).reshape(-1, n).sum(0) if uq.slot_relaxation else relaxed(theta)[0]
    top = torch.topk(w, 50).indices; inner = top[w[top] > torch.topk(w, 51).values[-1] + 0.02]
    dirn = torch.zeros(n, device=dev); dirn[inner] = torch.randn(inner.numel(), device=dev, generator=gen); dirn /= dirn.norm()
    th = theta.clone().requires_grad_(True)
    f = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, relaxed(th), target, differentiable=True)[0]
    gd = float(torch.autograd.grad(f, th)[0] @ dirn)
    with torch.no_grad():
        fd = {e: (uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, relaxed(theta + e * dirn), target, differentiable=True)[0].item()
                  - uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, relaxed(theta - e * dirn), target, differentiable=True)[0].item()) / (2 * e) for e in (1e-2, 1e-3)}
    rel = abs(gd - fd[1e-3]) / max(abs(fd[1e-3]), 1e-12)
    uq.y_range = y_range
    print(f"[{name}] 1. relaxed (binary / one-hot) vs hard update: max |diff| of E[Q] {d1:.1e} (E[Q] {hard.tolist()})", flush=True)
    print(f"[{name}] 2. gradient along the logits: autograd {gd:+.4e} vs finite difference {fd[1e-2]:+.4e} (1e-2), {fd[1e-3]:+.4e} (1e-3): rel. diff {rel:.1%}", flush=True)
    tol = 2e-2 if relax == "avg_emb" else 1e-4  # avg_emb: the batch's rows are not in the column blocks' context
    ok &= d1 < tol * hard.abs().max().item() and rel < 0.05
    # 3. risk objective
    if not a.skip_risk and relax != "avg_emb":
        g1 = torch.Generator(device=dev).manual_seed(5); g2 = torch.Generator(device=dev).manual_seed(5)
        with torch.no_grad():
            dh = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, Wb, 16, differentiable=False, generator=g1)
        dd = uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, Wr, 16, differentiable=True, generator=g2).detach()
        vh, vd = risk_and_estimate(target.q(dh))[0], risk_and_estimate(target.q(dd))[0]
        th2 = theta.clone().requires_grad_(True)
        rk = risk_and_estimate(target.q(uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, relaxed(th2), 16, differentiable=True,
                                                          generator=torch.Generator(device=dev).manual_seed(6))))[0].sum()
        gr = torch.autograd.grad(rk, th2)[0]
        print(f"[{name}] 3. risk: posterior variance of Q, hard draws {vh.tolist()} vs differentiable draws at binary weights {vd.tolist()}; "
              f"gradient through the draws finite {bool(torch.isfinite(gr).all())}, |grad| {gr.norm():.2e}", flush=True)
        ok &= bool(torch.isfinite(gr).all())
    # 4. cost per planner step
    torch.backends.cuda.matmul.allow_tf32 = True
    for obj in (() if a.skip_cost else ("gain", "gain_paired") + (() if (a.skip_risk or relax == "avg_emb") else ("risk",))):
        th3 = torch.full((n,), math.log(1.0 / n), device=dev, requires_grad=True)
        nz = gumbel_like(th3.detach().unsqueeze(0).expand(10, n), gen)
        torch.cuda.synchronize(); t0 = time.time(); torch.cuda.reset_peak_memory_stats(dev)
        for s0 in range(0, 10, 2):
            A = (soft_topk_slots(th3.unsqueeze(0).expand(2, n), 5, 0.5, noise=nz[s0:s0 + 2]) if uq.slot_relaxation
                 else soft_topk(th3.unsqueeze(0).expand(2, n), 5, 0.5, noise=nz[s0:s0 + 2], clamp_max=1.0))
            if obj == "risk":
                val = risk_and_estimate(target.q(uq.rollout_draws(st, prob.eval_x, prob.pool_x, ybar, A, 16, differentiable=True, generator=gen)))[0]
            else:
                val = uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar, A, target, differentiable=True)
                if obj == "gain_paired":
                    val = val - uq.rollout_target_mean(st, prob.eval_x, prob.pool_x, ybar.flip(0), A, target, differentiable=True)
            val.sum().backward()
        torch.cuda.synchronize()
        print(f"[{name}] 4. {obj:11s}: {time.time() - t0:.1f}s per planner step, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    del uq; torch.cuda.empty_cache()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
