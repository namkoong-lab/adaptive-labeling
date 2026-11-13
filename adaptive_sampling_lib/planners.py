"""One-step (E-KG) planners optimised with policy gradients, and active-learning baselines.

All planners see the same objects: a UQ module (see uq_gp / uq_ensemble / uq_tabpfn), a target
(targets.MSETarget / LogLossTarget) and the current problem state.  They return the indices
(into the current pool) of the batch to label.
"""
from __future__ import annotations

import math
import time

import torch

from .subset import soft_topk, soft_topk_slots, sample_hard_subset, ordered_logprob, gumbel_like
from .targets import risk_and_estimate


# --------------------------------------------------------------------------------------
# Active-learning baselines
# --------------------------------------------------------------------------------------
def select_random(n_pool: int, b: int, generator=None):
    return torch.randperm(n_pool, generator=generator)[:b]


def select_uncertainty_static(uq, state, pool_x, b: int):
    std = uq.pointwise_std(state, pool_x)
    return torch.topk(std, b).indices.cpu()


def select_uncertainty_sequential(uq, state, train_x, train_y, pool_x, b: int, generator=None):
    """Greedy batch: pick the most uncertain pool point, pseudo-label it with a draw from the
    current belief, condition, repeat (the paper's 'Uncertainty Sampling (Sequential)')."""
    pseudo = uq.sample_pool_labels(state, pool_x, 1, generator)[0]  # joint pseudo labels for the pool
    remaining = torch.arange(pool_x.shape[0])
    chosen = []
    tx, ty = train_x, train_y
    st = state
    for _ in range(b):
        std = uq.pointwise_std(st, pool_x[remaining])
        j = int(torch.argmax(std).item())
        idx = int(remaining[j].item())
        chosen.append(idx)
        tx = torch.cat([tx, pool_x[idx: idx + 1].to(tx)], 0)
        ty = torch.cat([ty, pseudo[idx: idx + 1].to(ty)], 0)
        st = uq.condition(tx, ty)
        remaining = remaining[remaining != idx]
    return torch.tensor(chosen)


# --------------------------------------------------------------------------------------
# Policy-gradient one-step planners
# --------------------------------------------------------------------------------------
class PolicyGradientPlanner:
    """Optimises item logits theta over the current pool to minimise the expected posterior risk
    after labelling a batch of size b (eq. (42)); then returns the top-b items.

    estimator = 'pathwise' : smoothed pathwise gradient through soft top-k + weighted update (eq. 45)
    estimator = 'reinforce': score-function gradient with baseline in {'none','mean','loo'}
    """

    def __init__(self, uq, target, b: int, n_steps: int, G: int, n_mc: int, lr: float = 0.1, tau: float = 0.1,
                 estimator: str = "pathwise", baseline: str = "none", grad_clip: float | None = 2.0,
                 imagined_labels: str = "per_step", log_every: int = 10, verbose: bool = False,
                 weight_decay: float = 0.0, init_logits: str = "uniform", objective: str = "risk", grad_chunk: int | None = None):
        self.grad_chunk = grad_chunk
        # objective = 'risk': rollout value is the MC posterior variance of the target after the update
        #             'gain': rollout value is -(M_+ - M_t)^2, the (negative) squared change of the
        #                     posterior mean of the target (Proposition 1); needs only posterior means
        #             'gain_paired': -(M_+(y) - M_+(y'))^2 / 2 with two independent imagined label vectors y, y' and the
        #                     same weights; its expectation is the label-driven variance of M_+ alone, free of the drift
        #                     E[M_+] - M_t that a non-martingale belief adds to 'gain' (two passes per rollout)
        self.objective = objective
        self.uq, self.target, self.b = uq, target, int(b)
        self.n_steps, self.G, self.n_mc, self.lr, self.tau = int(n_steps), int(G), int(n_mc), float(lr), float(tau)
        self.estimator, self.baseline, self.grad_clip = estimator, baseline, grad_clip
        self.imagined_labels, self.log_every, self.verbose, self.wd = imagined_labels, log_every, verbose, weight_decay
        self.init_logits = init_logits

    def _risk(self, state, eval_x, pool_x, ybar, W, differentiable):
        if self.objective == "gain_paired":
            yb2 = self._ybar_b[: W.shape[0]] if self._ybar_b.dim() == 2 and ybar.dim() == 2 else self._ybar_b
            m1 = self.uq.rollout_target_mean(state, eval_x, pool_x, ybar, W, self.target, differentiable=differentiable)
            m2 = self.uq.rollout_target_mean(state, eval_x, pool_x, yb2, W, self.target, differentiable=differentiable)
            return -0.5 * (m1 - m2) ** 2, m1
        if self.objective == "gain":
            m_plus = self.uq.rollout_target_mean(state, eval_x, pool_x, ybar, W, self.target, differentiable=differentiable)
            return -((m_plus - self._m_t) ** 2), m_plus
        draws = self.uq.rollout_draws(state, eval_x, pool_x, ybar, W, self.n_mc, differentiable=differentiable)
        Q = self.target.q(draws)  # [G, S]
        risk, est = risk_and_estimate(Q)
        return risk, est

    def select(self, state, pool_x, eval_x, generator=None):
        n = pool_x.shape[0]
        dev = pool_x.device
        theta = torch.full((n,), math.log(1.0 / n), device=dev, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=self.lr, weight_decay=self.wd)
        history = {"loss": [], "risk_mean": [], "grad_norm": [], "step_time": [], "topb": []}
        t_start = time.time()
        ybar = None
        if self.objective == "gain":
            with torch.no_grad():
                self._m_t = self.uq.target_mean(state, eval_x, self.target).detach()
            history["m_t"] = float(self._m_t)
        bank = None
        if self.imagined_labels == "per_step_bank":
            # all n_steps imagined label vectors of this epoch in one batched draw: the same distribution as one draw per
            # step (the belief does not change within select), much cheaper for the fully autoregressive sampler
            bank = self.uq.sample_pool_labels(state, pool_x, self.n_steps, generator)
        for it in range(self.n_steps):
            t0 = time.time()
            if bank is not None:
                ybar = bank[it]
            elif self.objective == "gain_paired":  # two independent imagined label vectors per step, shared by the rollouts
                assert self.imagined_labels == "per_step", "gain_paired uses one pair of imagined label vectors per step"
                pair = self.uq.sample_pool_labels(state, pool_x, 2, generator)
                ybar, self._ybar_b = pair[0], pair[1]
            elif ybar is None or self.imagined_labels in ("per_step", "per_rollout"):
                if self.imagined_labels == "per_rollout":
                    # G independent imagined label vectors per step, one per rollout ([G, N]): the per-step
                    # gradient then averages over G fantasies instead of using a single shared one
                    ybar = self.uq.sample_pool_labels(state, pool_x, self.G, generator)
                else:
                    ybar = self.uq.sample_pool_labels(state, pool_x, 1, generator)[0]  # shared across the G rollouts
            opt.zero_grad()
            if self.estimator == "pathwise":
                noise = gumbel_like(theta.detach().unsqueeze(0).expand(self.G, n), generator)
                gc = self.grad_chunk or self.G  # rollouts per backward pass (memory ~ gc, not G)
                risks = []
                for s0 in range(0, self.G, gc):
                    g_n = min(gc, self.G - s0)
                    if getattr(self.uq, "slot_relaxation", False):  # b relaxed one-hot slots [g_n, b, n] (average-row relaxations)
                        A = soft_topk_slots(theta.unsqueeze(0).expand(g_n, n), self.b, self.tau, noise=noise[s0: s0 + g_n])
                    else:
                        A = soft_topk(theta.unsqueeze(0).expand(g_n, n), self.b, self.tau, noise=noise[s0: s0 + g_n], clamp_max=1.0)  # [g_n, n]
                    yb = ybar[s0: s0 + g_n] if ybar.dim() == 2 else ybar  # per-rollout labels follow the chunk
                    risk_c, est = self._risk(state, eval_x, pool_x, yb, A, differentiable=True)
                    (risk_c.sum() / self.G).backward()
                    risks.append(risk_c.detach())
                risk = torch.cat(risks)
                loss = risk.mean()
            elif self.estimator == "reinforce":
                idxs, logps = [], []
                W = torch.zeros(self.G, n, device=dev)
                for g in range(self.G):
                    idx = sample_hard_subset(theta.detach(), self.b, generator)
                    idxs.append(idx)
                    logps.append(ordered_logprob(theta, idx))
                    W[g, idx] = 1.0
                with torch.no_grad():
                    risk, est = self._risk(state, eval_x, pool_x, ybar, W, differentiable=False)
                risk = risk.float().detach()
                logp = torch.stack(logps)
                if self.baseline == "none":
                    adv = risk
                elif self.baseline == "mean":
                    adv = risk - risk.mean()
                elif self.baseline == "loo":
                    adv = risk - (risk.sum() - risk) / max(self.G - 1, 1)
                else:
                    raise ValueError(self.baseline)
                loss = (adv * logp).mean()
                loss.backward()
            else:
                raise ValueError(self.estimator)
            gn = torch.nn.utils.clip_grad_norm_([theta], self.grad_clip) if self.grad_clip else theta.grad.norm()
            last_grad = theta.grad.detach().clone()
            opt.step()
            history["loss"].append(float(loss.item()))
            history["risk_mean"].append(float(risk.mean().item()))
            history["grad_norm"].append(float(gn))
            history["step_time"].append(time.time() - t0)
            if it % self.log_every == 0 or it == self.n_steps - 1:
                history["topb"].append(torch.topk(theta.detach(), self.b).indices.cpu().tolist())
                if self.verbose:
                    print(f"  [{self.estimator}] step {it:4d} risk={risk.mean().item():.4e} loss={loss.item():.4e} "
                          f"|g|={float(gn):.3e} dt={time.time() - t0:.2f}s top-b={history['topb'][-1]}", flush=True)
        history["total_time"] = time.time() - t_start
        history["final_logits"] = theta.detach().cpu()
        # top-b logits; exact ties (e.g. after a single Adam step, where every updated logit moved by
        # +-lr) are broken by the last policy gradient (more negative gradient = larger marginal gain)
        # (1e-3 << lr=0.1 keeps the primary ordering; float64 so the tie-break is not lost to rounding)
        score = theta.detach().double() - 1e-3 * last_grad.double() / (last_grad.abs().max().double() + 1e-30)
        # remaining exact ties (items with zero gradient, e.g. never sampled by REINFORCE) are broken at random
        score = score + 1e-9 * torch.rand(score.shape, device=score.device, dtype=score.dtype, generator=generator)
        return torch.topk(score, self.b).indices.cpu(), history
