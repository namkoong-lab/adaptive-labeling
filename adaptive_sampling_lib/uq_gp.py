"""Gaussian-process UQ module with the weighted posterior update of Algorithm 2
(port of the original GP pipeline's custom_gp_cholesky.py: off-diagonal entries of K scaled by
w_i w_j, rows of K_* scaled by w_i; binary weights give the ordinary GP posterior on the
selected observations).  Everything is done in float64 on the chosen device.
"""
from __future__ import annotations

import torch

from .data import rbf_kernel


class GPModule:
    name = "gp"

    def __init__(self, lengthscale: float, outputscale: float, noise_var: float, mean_constant: float = 0.0,
                 stabilizing: float = 1e-4, device="cpu", dtype=torch.float64):
        self.ls, self.os, self.nv, self.mc = float(lengthscale), float(outputscale), float(noise_var), float(mean_constant)
        self.stab = float(stabilizing)
        self.device, self.dtype = torch.device(device), dtype

    # ----- helpers -----
    def _t(self, x):
        return x.to(self.device, self.dtype)

    def kernel(self, a, b):
        return rbf_kernel(a, b, self.ls, self.os)

    def weighted_posterior(self, X, y, w, Xq):
        """mu [.., Nq], cov [.., Nq, Nq] of the latent f at Xq given weighted observations (X, y, w).
        Batched: X [.., n, d] (or [n, d] shared), y [.., n] (or [n]), w [.., n]; Xq [Nq, d]."""
        n = w.shape[-1]
        I = torch.eye(n, device=Xq.device, dtype=Xq.dtype)
        K = self.kernel(X, X) + (self.nv + self.stab) * I  # [.., n, n]
        W = w.unsqueeze(-1) * w.unsqueeze(-2)
        Kw = K * (W * (1 - I) + I)  # weights only on the off-diagonal
        Ks = w.unsqueeze(-1) * self.kernel(X, Xq)  # [.., n, Nq]
        Kss = self.kernel(Xq, Xq) + self.stab * torch.eye(Xq.shape[0], device=Xq.device, dtype=Xq.dtype)
        L = torch.linalg.cholesky(Kw)
        yc = (y - self.mc).unsqueeze(-1)
        if yc.dim() < L.dim():
            yc = yc.expand(*L.shape[:-1], 1)
        alpha = torch.cholesky_solve(yc, L)
        mu = self.mc + (Ks.transpose(-1, -2) @ alpha).squeeze(-1)
        v = torch.linalg.solve_triangular(L, Ks, upper=False)
        cov = Kss - v.transpose(-1, -2) @ v
        return mu, cov

    @staticmethod
    def _draw(mu, cov, n, generator=None, jitter=1e-5):
        """mu [.., N], cov [.., N, N] -> draws [.., n, N]."""
        L = torch.linalg.cholesky(cov + jitter * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype))
        z = torch.randn(*mu.shape[:-1], n, mu.shape[-1], device=mu.device, dtype=mu.dtype, generator=generator)
        return mu.unsqueeze(-2) + z @ L.transpose(-1, -2)

    # ----- UQ interface -----
    def condition(self, train_x, train_y):
        return {"X": self._t(train_x), "y": self._t(train_y)}

    def posterior_draws(self, state, eval_x, n_draws, generator=None):
        X, y = state["X"], state["y"]
        mu, cov = self.weighted_posterior(X, y, torch.ones(X.shape[0], device=X.device, dtype=X.dtype), self._t(eval_x))
        return self._draw(mu, cov, n_draws, generator)  # [S, N_eval]

    def posterior_mean_std(self, state, xq):
        X, y = state["X"], state["y"]
        mu, cov = self.weighted_posterior(X, y, torch.ones(X.shape[0], device=X.device, dtype=X.dtype), self._t(xq))
        return mu, cov.diagonal().clamp_min(0).sqrt()

    def sample_pool_labels(self, state, pool_x, n=1, generator=None):
        """Joint draw of noisy labels for all pool units from the current posterior predictive."""
        X, y = state["X"], state["y"]
        P = self._t(pool_x)
        mu, cov = self.weighted_posterior(X, y, torch.ones(X.shape[0], device=X.device, dtype=X.dtype), P)
        cov = cov + self.nv * torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
        return self._draw(mu, cov, n, generator)  # [n, N_pool]

    def pointwise_std(self, state, pool_x):
        return self.posterior_mean_std(state, pool_x)[1]

    def target_mean(self, state, eval_x, target):
        """Closed-form posterior mean of the target, E[Q | D] (no Monte Carlo)."""
        mu, std = self.posterior_mean_std(state, eval_x)
        return target.mean_from_moments(mu, std ** 2)

    def _rollout_inputs(self, state, extra_x, extra_y, weights, differentiable):
        """Batched (X [G, n, d], y [G, n], w [G, n]) for the G rollouts: hard rollouts gather the selected
        items (equal batch sizes), soft rollouts use all pool items with their weights."""
        X, y = state["X"], state["y"]
        Px, Py = self._t(extra_x), self._t(extra_y)
        W = weights.to(self.device, self.dtype)
        G = W.shape[0]
        ones = torch.ones(G, X.shape[0], device=X.device, dtype=X.dtype)
        if Py.dim() == 1:
            Py = Py.unsqueeze(0).expand(G, -1)
        if not differentiable:
            sel = torch.stack([(W[g] > 0).nonzero().squeeze(1) for g in range(G)])  # [G, b]
            Xg = torch.cat([X.unsqueeze(0).expand(G, -1, -1), Px[sel]], 1)
            yg = torch.cat([y.unsqueeze(0).expand(G, -1), torch.gather(Py, 1, sel)], 1)
            wg = torch.cat([ones, torch.gather(W, 1, sel)], 1)
        else:
            Xg = torch.cat([X.unsqueeze(0).expand(G, -1, -1), Px.unsqueeze(0).expand(G, -1, -1)], 1)
            yg = torch.cat([y.unsqueeze(0).expand(G, -1), Py], 1)
            wg = torch.cat([ones, W], 1)
        return Xg, yg, wg

    def rollout_target_mean(self, state, eval_x, extra_x, extra_y, weights, target, differentiable=False):
        """E[Q | D, weighted pseudo-update] for each rollout: [G] (closed form, batched over rollouts)."""
        Xg, yg, wg = self._rollout_inputs(state, extra_x, extra_y, weights, differentiable)
        mu, cov = self.weighted_posterior(Xg, yg, wg, self._t(eval_x))
        return target.mean_from_moments(mu, cov.diagonal(dim1=-2, dim2=-1).clamp_min(0))

    def rollout_draws(self, state, eval_x, extra_x, extra_y, weights, n_draws, differentiable=False, generator=None):
        """Posterior draws at eval_x after the weighted pseudo-update with (extra_x, extra_y, weights).
        extra_y: [N_extra] or [G, N_extra]; weights: [G, N_extra] (in [0,1]).  Returns [G, S, N_eval].
        All G rollouts are processed as one batched Cholesky."""
        Xg, yg, wg = self._rollout_inputs(state, extra_x, extra_y, weights, differentiable)
        mu, cov = self.weighted_posterior(Xg, yg, wg, self._t(eval_x))
        return self._draw(mu, cov, n_draws, generator)
