"""Ensemble+ (randomized-prior deep ensemble) UQ module, fully vectorized.

Each member m combines a trainable MLP g_m with a fixed, randomly initialised prior MLP p_m:
f_m(x) = g_m(x) + alpha * p_m(x).  The posterior update is a fixed number of (weighted) Adam
steps on  sum_i w_i (f_m(x_i) - y_i)^2 + lambda ||eta_m||^2  starting from the member's initial
parameters (as in the original pipeline).  All members (and all G rollouts) are trained at once
as one batched computation: parameters have a leading "replica" dimension R = G * M and the
MLP is written with batched matmuls, so the inner loop is ~50 small kernel launches per step
instead of a Python loop over members.  The unrolled Adam update is written in plain torch, so
it is differentiable w.r.t. the observation weights (create_graph=True) when `differentiable`.
This replaces torchopt's MetaAdam with no library dependency.
"""
from __future__ import annotations

import math

import torch


def _linear_init(R, fan_in, fan_out, generator, device, dtype):
    bound = 1.0 / math.sqrt(fan_in)
    W = (torch.rand(R, fan_in, fan_out, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound
    b = (torch.rand(R, 1, fan_out, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound
    return W, b


def _xavier_init(R, fan_in, fan_out, generator, device, dtype):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    W = (torch.rand(R, fan_in, fan_out, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound
    b = torch.zeros(R, 1, fan_out, device=device, dtype=dtype)
    return W, b


_ACTS = {"relu": torch.relu, "silu": torch.nn.functional.silu, "tanh": torch.tanh, "softplus": torch.nn.functional.softplus,
         "gelu": torch.nn.functional.gelu}


def batched_mlp(params: dict, x: torch.Tensor, n_layers: int, activation: str = "relu"):
    """params: {W0,b0,...}: W_l [R, in, out], b_l [R, 1, out].  x: [N, d] or [R, N, d] -> [R, N, out]."""
    act = _ACTS[activation]
    h = x if x.dim() == 3 else x.unsqueeze(0).expand(params["W0"].shape[0], -1, -1)
    for l in range(n_layers):
        h = torch.baddbmm(params[f"b{l}"], h, params[f"W{l}"])
        if l < n_layers - 1:
            h = act(h)
    return h


class EnsemblePlusModule:
    name = "ensemble"

    def __init__(self, input_dim: int, hidden_sizes=(50, 50), n_members: int = 10, alpha: float | str = "auto",
                 prior_target_std: float = 0.8, n_inner_steps: int = 50, inner_lr: float = 0.1, weight_decay: float = 0.1,
                 noise_std: float = 0.1, seed_base: int = 0, seed_prior: int = 0, device="cpu", dtype=torch.float32,
                 n_classes: int = 1, calib_x: torch.Tensor | None = None, anchor_to_init: bool = True,
                 ridge: float = 1e-4, adam_eps: float = 1e-8, inner_opt: str = "adam", grad_last_k: int | None = None,
                 momentum: float = 0.9, activation: str = "relu", lr_per_point: bool = False):
        # lr_per_point: inner learning rate = inner_lr / (effective number of weighted observations), which
        # keeps (heavy-ball) gradient descent on the *sum* loss stable as the context grows
        self.activation, self.lr_per_point = activation, bool(lr_per_point)
        self.anchor_to_init = bool(anchor_to_init)
        self.ridge, self.adam_eps = float(ridge), float(adam_eps)
        self.inner_opt, self.grad_last_k, self.momentum = inner_opt, grad_last_k, float(momentum)
        self.d, self.hidden, self.M = int(input_dim), tuple(hidden_sizes), int(n_members)
        self.n_steps, self.lr, self.wd, self.noise_std = int(n_inner_steps), float(inner_lr), float(weight_decay), float(noise_std)
        self.device, self.dtype, self.C = torch.device(device), dtype, int(n_classes)
        sizes = [self.d] + list(self.hidden) + [self.C]
        self.sizes = sizes
        self.n_layers = len(sizes) - 1
        gb = torch.Generator(device="cpu").manual_seed(seed_base)
        gp = torch.Generator(device="cpu").manual_seed(seed_prior)
        self.init_params = {}
        self.prior_params = {}
        for l in range(self.n_layers):
            W, b = _linear_init(self.M, sizes[l], sizes[l + 1], gb, "cpu", dtype)
            self.init_params[f"W{l}"], self.init_params[f"b{l}"] = W.to(self.device), b.to(self.device)
            Wp, bp = _xavier_init(self.M, sizes[l], sizes[l + 1], gp, "cpu", dtype)
            self.prior_params[f"W{l}"], self.prior_params[f"b{l}"] = Wp.to(self.device), bp.to(self.device)
        if alpha == "auto":
            assert calib_x is not None, "alpha='auto' needs calib_x"
            with torch.no_grad():
                p = self.prior(calib_x.to(self.device, dtype))  # [M, N, C]
                emp = p.std(dim=0).mean().item()
            self.alpha = float(prior_target_std / max(emp, 1e-8))
        else:
            self.alpha = float(alpha)

    # ----- forward passes -----
    def prior(self, x):
        return batched_mlp(self.prior_params, x, self.n_layers, self.activation)  # [M, N, C]

    def _tile(self, params, G):
        return {k: v.repeat(G, *([1] * (v.dim() - 1))) for k, v in params.items()}

    def predict(self, params, x, G=1):
        """f(x) for all replicas.  Returns [G, M, N, C]."""
        base = batched_mlp(params, x, self.n_layers, self.activation)  # [G*M, N, C]
        pri = self.prior(x)  # [M, N, C]
        out = base.view(G, self.M, *base.shape[1:]) + self.alpha * pri.unsqueeze(0)
        return out

    # ----- inner training (the posterior update) -----
    def fit(self, X, y, w=None, G=1, differentiable=False):
        """Train G*M replicas from the shared initial parameters on (X, y) with per-replica-group
        observation weights w [G, N] (None -> ones).  y: [N] (regression) or [N] long (classification).
        Returns the trained params dict with leading dim G*M."""
        X = X.to(self.device, self.dtype)
        N = X.shape[0]
        if w is None:
            w = torch.ones(G, N, device=self.device, dtype=self.dtype)
        w = w.to(self.device, self.dtype)
        wR = w.unsqueeze(1).expand(G, self.M, N).reshape(G * self.M, N, 1)  # [R, N, 1]
        init_tiled = self._tile(self.init_params, G)
        params = {k: v.detach().clone().requires_grad_(True) for k, v in init_tiled.items()}
        pri = self.prior(X).repeat(G, 1, 1)  # [R, N, C]
        if self.C == 1:
            yR = y.to(self.device, self.dtype).view(1, N, 1)
        else:
            yR = y.to(self.device).long().view(1, N).expand(G * self.M, N)
        m = {k: torch.zeros_like(v) for k, v in params.items()}
        v2 = {k: torch.zeros_like(v) for k, v in params.items()}
        b1, b2, eps = 0.9, 0.999, self.adam_eps
        keys = list(params.keys())
        lr = self.lr / wR.detach().sum(1).clamp_min(1.0).view(-1, 1, 1) if self.lr_per_point else self.lr  # [R,1,1] or scalar
        ctx = torch.enable_grad()
        with ctx:
            for t in range(1, self.n_steps + 1):
                # truncated unrolled differentiation: only the last `grad_last_k` steps carry the graph
                track = differentiable and (self.grad_last_k is None or t > self.n_steps - self.grad_last_k)
                out = batched_mlp(params, X, self.n_layers, self.activation) + self.alpha * pri  # [R, N, C]
                if self.C == 1:
                    data_loss = (wR * (out - yR) ** 2).sum()
                else:
                    nll = torch.nn.functional.cross_entropy(out.reshape(-1, self.C), yR.reshape(-1), reduction="none").view(G * self.M, N, 1)
                    data_loss = (wR * nll).sum()
                if self.anchor_to_init:  # anchored ensemble: regularise towards the (random) initialisation
                    reg = sum(((params[k] - init_tiled[k]) ** 2).sum() for k in keys) * self.wd
                    # small ridge towards zero: keeps every coordinate's gradient (and hence Adam's second
                    # moment) away from exactly zero, which otherwise makes the unrolled derivative ~1/eps
                    reg = reg + sum((params[k] ** 2).sum() for k in keys) * self.ridge
                else:
                    reg = sum((params[k] ** 2).sum() for k in keys) * self.wd
                loss = data_loss + reg
                grads = torch.autograd.grad(loss, [params[k] for k in keys], create_graph=track)
                new_params = {}
                for k, g in zip(keys, grads):
                    if self.inner_opt != "adam":  # (heavy-ball) gradient descent: linear, well-conditioned unroll
                        m[k] = self.momentum * m[k] + g if self.inner_opt == "momentum" else g
                        upd = params[k] - lr * m[k]
                        new_params[k] = upd if track else upd.detach().requires_grad_(True)
                        continue
                    m[k] = b1 * m[k] + (1 - b1) * g
                    # The second-moment normaliser is treated as a constant in the backward pass
                    # (stop-gradient): its derivative w.r.t. a coordinate with (near-)zero gradient is
                    # ~1/eps and compounds over the unrolled steps into inf/NaN pathwise gradients (this
                    # happens e.g. for anchored members whose dead units receive exactly zero gradient).
                    # The pathwise signal flows through the gradients and the momentum.
                    v2[k] = b2 * v2[k] + (1 - b2) * (g.detach() ** 2)
                    mhat = m[k] / (1 - b1 ** t)
                    vhat = v2[k] / (1 - b2 ** t)
                    upd = params[k] - lr * mhat / (vhat.sqrt() + eps)
                    new_params[k] = upd if track else upd.detach().requires_grad_(True)
                params = new_params
        if not differentiable:
            params = {k: v.detach() for k, v in params.items()}
        return params

    # ----- UQ interface -----
    def condition(self, train_x, train_y):
        params = self.fit(train_x, train_y, G=1, differentiable=False)
        return {"X": train_x.to(self.device, self.dtype), "y": train_y.to(self.device), "params": params}

    def _members_at(self, params, x, G=1):
        out = self.predict(params, x.to(self.device, self.dtype), G)  # [G, M, N, C]
        if self.C == 1:
            return out.squeeze(-1)  # [G, M, N]
        return torch.softmax(out, dim=-1)  # [G, M, N, C] class probabilities

    def posterior_draws(self, state, eval_x, n_draws=None, generator=None):
        with torch.no_grad():
            return self._members_at(state["params"], eval_x, 1)[0]  # [M, N(,C)]

    def posterior_mean_std(self, state, xq):
        with torch.no_grad():
            f = self._members_at(state["params"], xq, 1)[0]
        if self.C == 1:
            return f.mean(0), f.std(0)
        return f.mean(0), f.std(0).sum(-1)

    def pointwise_std(self, state, pool_x):
        return self.posterior_mean_std(state, pool_x)[1]

    def target_mean(self, state, eval_x, target):
        return target.q(self.posterior_draws(state, eval_x)).mean(-1)

    def rollout_target_mean(self, state, eval_x, extra_x, extra_y, weights, target, differentiable=False):
        draws = self.rollout_draws(state, eval_x, extra_x, extra_y, weights, None, differentiable)
        return target.q(draws).mean(-1)  # [G]

    def sample_pool_labels(self, state, pool_x, n=1, generator=None):
        with torch.no_grad():
            f = self._members_at(state["params"], pool_x, 1)[0]  # [M, N(,C)]
            gdev = generator.device if generator is not None else torch.device("cpu")
            idx = torch.randint(0, self.M, (n,), generator=generator, device=gdev).to(f.device)
            if self.C == 1:
                noise = torch.randn(n, f.shape[1], device=gdev, dtype=f.dtype, generator=generator).to(f.device)
                return f[idx] + self.noise_std * noise
            probs = f[idx]  # [n, N, C]
            return torch.distributions.Categorical(probs=probs).sample()  # [n, N] long

    def rollout_draws(self, state, eval_x, extra_x, extra_y, weights, n_draws=None, differentiable=False, generator=None):
        """Retrain all G*M replicas from init on train + extra with weights [G, N_extra]; return
        member predictions at eval_x: [G, M, N_eval(, C)]."""
        X = torch.cat([state["X"], extra_x.to(self.device, self.dtype)], 0)
        ey = extra_y if extra_y.dim() == 1 else extra_y[0]
        if extra_y.dim() == 2 and extra_y.shape[0] > 1:
            raise NotImplementedError("per-rollout imagined labels not supported for the ensemble module (use shared labels)")
        y = torch.cat([state["y"].to(ey), ey.to(self.device)], 0)
        G = weights.shape[0]
        w = torch.cat([torch.ones(G, state["X"].shape[0], device=self.device, dtype=self.dtype), weights.to(self.device, self.dtype)], 1)
        params = self.fit(X, y, w, G=G, differentiable=differentiable)
        if differentiable:
            return self._members_at(params, eval_x, G)
        with torch.no_grad():
            return self._members_at(params, eval_x, G)
