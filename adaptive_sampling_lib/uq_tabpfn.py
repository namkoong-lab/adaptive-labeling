"""TabPFN (v2.5 architecture) as an uncertainty-quantification module for the planner.

Posterior state.  The belief is TabPFN's in-context predictive given the labelled data D.
Joint posterior draws of the target over the evaluation set are produced by *chunked
autoregressive fantasy sampling*: the evaluation points are visited in K random chunks; the
labels of chunk k are sampled from TabPFN's predictive given D plus the fantasised chunks
< k, and appended to the context.  For regression the bar-distribution sample is produced
by a reparameterised inverse-CDF (differentiable in the logits); for classification by a
straight-through Gumbel-softmax.

Weighted (fractional) pseudo-update.  The planner's relaxed batch assigns a weight
a_i in [0,1] to every pool item.  We realise the weighted update inside the transformer:
in every between-rows (column) attention layer a context row i receives the additive
attention-logit bias log a_i for every query (so a_i = 1 is the ordinary update, a_i = 0
removes the row exactly), and the feature standardisation / target-imputation statistics
that TabPFN computes over the context rows are replaced by their a-weighted versions.  The
rollout objective is therefore a differentiable function of the weights, which gives the
pathwise policy gradient; binary weights reproduce the ordinary hard update exactly.

Only the v2.5 architecture is patched (tabpfn.architectures.tabpfn_v2_5); the model's
weights are frozen.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from adaptive_sampling_lib.paths import TABPFN_MODELS  # noqa: E402

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", TABPFN_MODELS)

from tabpfn.architectures import tabpfn_v2_5 as v25  # noqa: E402
from tabpfn.architectures.interface import PerformanceOptions  # noqa: E402
from tabpfn.model_loading import load_model_criterion_config  # noqa: E402

# --------------------------------------------------------------------------------------
# Patches for the weighted context
# --------------------------------------------------------------------------------------
_WCTX = {"w": None}  # [N_ctx, B] context weights (no thinking rows) during a weighted forward
_ORIG_COL_FORWARD = v25.AlongColumnAttention.forward
_ORIG_IMPUTE = v25._impute_nan_and_inf_with_mean


def _expand_w(w, like_dim1):
    """w: [Rt, B] -> [Rt, like_dim1] where like_dim1 = B * k (flattened (B, k) order)."""
    Rt, B = w.shape
    k = like_dim1 // B
    assert k * B == like_dim1, (w.shape, like_dim1)
    return w[:, :, None].expand(Rt, B, k).reshape(Rt, like_dim1)


def _weighted_col_attention_forward(self, x_BcRE, single_eval_pos=None, *, cached_kv=None, return_kv=False):
    logw = getattr(self, "_ctx_logw", None)  # [B, N] incl. thinking rows
    if logw is None or cached_kv is not None or return_kv:
        return _ORIG_COL_FORWARD(self, x_BcRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)
    Bc, R, _ = x_BcRE.shape
    N = R if single_eval_pos is None else single_eval_pos
    B = logw.shape[0]
    C = Bc // B
    assert C * B == Bc and logw.shape[1] == N, (logw.shape, Bc, N)
    q = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)
    k = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    v = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
    bias = logw.to(q.dtype)[:, None, :].expand(B, C, N).reshape(Bc, 1, 1, N)

    def sdpa(qq, kk, vv):
        H = qq.shape[2]
        if kk.shape[2] != H:
            kk = kk.expand(-1, -1, H, -1)
            vv = vv.expand(-1, -1, H, -1)
        out = F.scaled_dot_product_attention(qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), attn_mask=bias)
        return out.transpose(1, 2)

    if single_eval_pos == R or single_eval_pos is None:
        out = sdpa(q, k, v)
    else:
        out = torch.cat([sdpa(q[:, :N], k, v), sdpa(q[:, N:], k[:, :, :1], v[:, :, :1])], dim=1)
    out = out.reshape(Bc, R, self.head_dim * self.num_heads)
    return self.out_projection(out), None


def _weighted_impute(x, num_train_rows, scaler_cache=None):
    w = _WCTX["w"]
    if w is None or scaler_cache is not None:
        return _ORIG_IMPUTE(x, num_train_rows, scaler_cache)
    xt = x[:num_train_rows]
    ww = _expand_w(w.to(x.dtype), xt.shape[1])
    while ww.dim() < xt.dim():
        ww = ww.unsqueeze(-1)
    finite = torch.isfinite(xt)
    ww = ww * finite
    means = (torch.where(finite, xt, torch.zeros_like(xt)) * ww).sum(0) / ww.sum(0).clamp_min(1e-12)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    return torch.where(nan_mask, means.unsqueeze(0).expand_as(x), x), nan_mask


class _WeightedScaler:
    """Drop-in for TorchStandardScaler whose fit uses the context weights when set."""

    def __init__(self, orig):
        self.orig = orig

    def fit(self, x):
        w = _WCTX["w"]
        if w is None:
            return self.orig.fit(x)
        ww = _expand_w(w.to(x.dtype), x.shape[1]).unsqueeze(-1)  # [Rt, Bg, 1]
        s = ww.sum(0).clamp_min(1e-12)
        mean = (ww * x).sum(0) / s
        # (sum w - 1) mirrors torch_nanstd's N-1 correction for binary weights
        var = (ww * (x - mean) ** 2).sum(0) / (s - 1.0).clamp_min(1.0)
        const = var < 1e-12  # constant (e.g. zero-padded) columns: unit scale, no sqrt(0) gradient
        std = torch.where(const, torch.ones_like(var), var.clamp_min(1e-12)).sqrt()
        std = torch.where(const, torch.ones_like(std), std)
        return {"mean": mean, "std": std}

    def transform(self, x, fitted_cache):
        return self.orig.transform(x, fitted_cache)

    def __call__(self, x, num_train_rows=None):
        return self.orig(x, num_train_rows)


def install_patches():
    v25.AlongColumnAttention.forward = _weighted_col_attention_forward
    v25._impute_nan_and_inf_with_mean = _weighted_impute


install_patches()


# --------------------------------------------------------------------------------------
# Bar-distribution helpers (reparameterised sampling)
# --------------------------------------------------------------------------------------
def bar_icdf(bardist, logits, u):
    """Inverse CDF of a FullSupportBarDistribution at per-element probabilities u (same leading
    shape as logits[..., 0]); differentiable w.r.t. logits.  Tails use the half-normal law."""
    probs = torch.softmax(logits.float(), -1)
    cum = torch.cumsum(probs, -1)
    idx = torch.searchsorted(cum, u.unsqueeze(-1).to(cum)).squeeze(-1).clamp(0, probs.shape[-1] - 1)
    cum0 = torch.cat([torch.zeros_like(cum[..., :1]), cum], -1)
    p_idx = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    rest = (u - cum0.gather(-1, idx.unsqueeze(-1)).squeeze(-1)) / p_idx
    rest = rest.clamp(1e-4, 1 - 1e-4)
    borders = bardist.borders.to(probs)
    widths = bardist.bucket_widths.to(probs)
    y_mid = borders[idx] + widths[idx] * rest
    hn_l = bardist.halfnormal_with_p_weight_before(widths[0])
    hn_r = bardist.halfnormal_with_p_weight_before(widths[-1])
    y_left = borders[1] - hn_l.icdf(1 - rest)
    y_right = borders[-2] + hn_r.icdf(rest)
    nb = probs.shape[-1]
    return torch.where(idx == 0, y_left, torch.where(idx == nb - 1, y_right, y_mid))


def bar_icdf_grid(bardist, logits, u):
    """bar_icdf evaluated at a shared grid of quantile levels u [K] -> [..., K] (probs/cumsum computed once).
    Differentiable w.r.t. logits (through the bin probabilities)."""
    probs = torch.softmax(logits.float(), -1)
    cum = torch.cumsum(probs, -1)
    nb = probs.shape[-1]
    uu = u.to(cum).view(*([1] * (cum.dim() - 1)), -1).expand(*cum.shape[:-1], -1)  # [..., K]
    idx = torch.searchsorted(cum, uu).clamp(0, nb - 1)
    cum0 = torch.cat([torch.zeros_like(cum[..., :1]), cum], -1)
    p_idx = probs.gather(-1, idx).clamp_min(1e-12)
    rest = ((uu - cum0.gather(-1, idx)) / p_idx).clamp(1e-4, 1 - 1e-4)
    borders = bardist.borders.to(probs)
    widths = bardist.bucket_widths.to(probs)
    y_mid = borders[idx] + widths[idx] * rest
    hn_l = bardist.halfnormal_with_p_weight_before(widths[0])
    hn_r = bardist.halfnormal_with_p_weight_before(widths[-1])
    y_left = borders[1] - hn_l.icdf(1 - rest)
    y_right = borders[-2] + hn_r.icdf(rest)
    return torch.where(idx == 0, y_left, torch.where(idx == nb - 1, y_right, y_mid))


def _draw_seed(generator):
    dev = generator.device if generator is not None else torch.device("cpu")
    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=dev).item())


# --------------------------------------------------------------------------------------
# The UQ module
# --------------------------------------------------------------------------------------
class TabPFNModule:
    name = "tabpfn"

    def __init__(self, task: str = "regression", version: str = "v2.5", device="cuda", n_draws: int = 32,
                 n_chunks: int = 4, n_pool_chunks: int = 5, autocast: bool = True, checkpoint: bool = True,
                 draw_type: str | None = None, soft_topm: int | None = None, n_classes: int = 2,
                 max_batch_rows: int = 40_000, u_clip: float = 0.005, y_range: tuple | None = None, n_quantiles: int = 128,
                 input_transform: str = "none"):
        assert version == "v2.5", "weighted context is implemented for the v2.5 architecture"
        self.task, self.device = task, torch.device(device)
        self.n_draws, self.n_chunks, self.n_pool_chunks = int(n_draws), int(n_chunks), int(n_pool_chunks)
        self.autocast, self.checkpoint, self.soft_topm = autocast, checkpoint, soft_topm
        self.n_classes = int(n_classes)
        self.draw_type = draw_type or ("labels" if task == "regression" else "mean")
        self.max_batch_rows = int(max_batch_rows)
        self.u_clip = float(u_clip)
        # y_range (raw label units): if given, regression moments (mean/variance for the target and the pointwise
        # std) are those of the predictive *conditioned on the label lying in this range*, computed on a grid of
        # n_quantiles quantile levels; fantasy label draws are clamped to it.  Removes the heavy half-normal tails
        # of the bar distribution that otherwise dominate the squared-loss target far from the context.
        self.y_range = None if y_range is None else (float(y_range[0]), float(y_range[1]))
        self.n_quantiles = int(n_quantiles)
        # feature transform fitted on each state's labelled points and applied to every feature that enters TabPFN
        # ('none' | 'fix': a feature constant among the labelled points is shifted to exactly 0 there, so TabPFN's zero-spread
        # rule applies | 'squash3' / 'squash10': (x - median) / IQR then z / sqrt(1 + (z/B)^2) | 'asinh'); fixed within an epoch,
        # so the planner's candidate batches cannot change it
        assert input_transform in ("none", "fix", "squash3", "squash10", "asinh"), input_transform
        self.input_transform = input_transform
        est = "regressor" if task == "regression" else "classifier"
        models, crit, cfgs, inf_cfg = load_model_criterion_config(
            model_path=None, check_bar_distribution_criterion=(task == "regression"), estimator_type=est,
            version="v2.5", download_if_not_exists=True)
        self.model = models[0].to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert isinstance(self.model, v25.TabPFNV2p5), type(self.model)
        self.model.standard_scaler = _WeightedScaler(self.model.standard_scaler)
        self.bardist = crit.to(self.device) if task == "regression" else None
        self.n_thinking = self.model.add_thinking_rows.num_thinking_rows
        self.perf = PerformanceOptions(save_peak_memory_factor=None, force_recompute_layer=bool(checkpoint))
        self.perf_nograd = PerformanceOptions(save_peak_memory_factor=None, force_recompute_layer=False)

    # ----- raw forward -----
    def _set_weights(self, w_NB):
        """w_NB: [N, B] context weights (None clears)."""
        if w_NB is None:
            _WCTX["w"] = None
            for m in self.model.modules():
                if isinstance(m, v25.AlongColumnAttention):
                    m._ctx_logw = None
            return
        _WCTX["w"] = w_NB
        logw = torch.log(w_NB.clamp_min(1e-30)).t()  # [B, N]
        logw = torch.cat([torch.zeros(logw.shape[0], self.n_thinking, device=logw.device, dtype=logw.dtype), logw], 1)
        for m in self.model.modules():
            if isinstance(m, v25.AlongColumnAttention):
                m._ctx_logw = logw

    def forward_raw(self, ctx_x, ctx_y, query_x, ctx_w=None, grad=False):
        """ctx_x: [N, C] or [B, N, C]; ctx_y: [N] or [N, B] (float; z-normed for regression, class
        index for classification); query_x: [M, C] or [B, M, C]; ctx_w: [N, B] or None.
        Returns logits [B, M, n_out]."""
        if ctx_y.dim() == 1:
            ctx_y = ctx_y.unsqueeze(1)
        N, B = ctx_y.shape
        cx = ctx_x if ctx_x.dim() == 3 else ctx_x.unsqueeze(0).expand(B, -1, -1)
        qx = query_x if query_x.dim() == 3 else query_x.unsqueeze(0).expand(B, -1, -1)
        M = qx.shape[1]
        x = torch.cat([cx, qx], 1).transpose(0, 1).contiguous().to(self.device, torch.float32)  # [N+M, B, C]
        y = ctx_y.contiguous().to(self.device, torch.float32)  # [N, B]
        w = None if ctx_w is None else ctx_w.to(self.device, torch.float32)

        def _run(x_, y_, w_):
            # the weights are (re)installed inside so that checkpoint recomputation sees them
            self._set_weights(w_)
            # mixed precision: bfloat16 by default (attribute autocast_dtype, e.g. torch.float16 = TabPFN's own GPU default); autocast=False runs float32
            with torch.autocast(device_type="cuda", dtype=getattr(self, "autocast_dtype", torch.bfloat16), enabled=(self.autocast and self.device.type == "cuda")):
                return self.model(x_, y_, only_return_standard_out=True, performance_options=self.perf_nograd)

        if grad:
            with torch.enable_grad():
                if self.checkpoint:
                    # whole-forward activation checkpointing (per-layer checkpointing cannot see the
                    # per-call weights at recomputation time)
                    out = torch.utils.checkpoint.checkpoint(_run, x, y, w, use_reentrant=False)
                else:
                    out = _run(x, y, w)
        else:
            with torch.no_grad():
                out = _run(x, y, w)
        return out.float().transpose(0, 1)  # [B, M, n_out]

    def _batched_forward(self, ctx_x, ctx_y, query_x, ctx_w, grad):
        """Split the batch so that batch * rows stays below max_batch_rows (memory)."""
        N, B = ctx_y.shape if ctx_y.dim() == 2 else (ctx_y.shape[0], 1)
        M = query_x.shape[-2]
        per = max(1, self.max_batch_rows // (N + M + self.n_thinking))
        if B <= per or grad:
            return self.forward_raw(ctx_x, ctx_y, query_x, ctx_w, grad)
        outs = []
        for s in range(0, B, per):
            sl = slice(s, min(B, s + per))
            cx = ctx_x if ctx_x.dim() == 2 else ctx_x[sl]
            qx = query_x if query_x.dim() == 2 else query_x[sl]
            outs.append(self.forward_raw(cx, ctx_y[:, sl], qx, None if ctx_w is None else ctx_w[:, sl], grad))
        return torch.cat(outs, 0)

    # ----- target space helpers -----
    def _znorm(self, y, state):
        return (y - state["y_mean"]) / state["y_std"]

    def _deznorm(self, z, state):
        return z * state["y_std"] + state["y_mean"]

    def _moments(self, logits, state):
        """Regression predictive mean and variance in raw label units: full bar-distribution moments, or
        moments conditioned on self.y_range (quantile-grid truncation)."""
        if self.y_range is None:
            m = self._deznorm(self.bardist.mean(logits), state)
            v = self.bardist.variance(logits) * state["y_std"] ** 2
            return m, v.clamp_min(0)
        K = self.n_quantiles
        u = (torch.arange(K, device=logits.device, dtype=torch.float32) + 0.5) / K
        z = bar_icdf_grid(self.bardist, logits, u)  # [..., K] (z-space)
        zlo, zhi = (self._znorm(torch.tensor(self.y_range[0], device=logits.device), state),
                    self._znorm(torch.tensor(self.y_range[1], device=logits.device), state))
        inside = ((z >= zlo) & (z <= zhi)).to(z)
        n_in = inside.sum(-1)
        zc = z.clamp(zlo, zhi)  # fallback (all mass outside): clamped values
        w = torch.where(n_in[..., None] > 0, inside, torch.ones_like(inside))
        zz = torch.where(n_in[..., None] > 0, z, zc)
        n = w.sum(-1)
        m = (zz * w).sum(-1) / n
        v = (((zz - m[..., None]) ** 2) * w).sum(-1) / n
        return self._deznorm(m, state), v * state["y_std"] ** 2

    def _sample_from_logits(self, logits, generator, differentiable):
        """logits [B, M, n_out] -> label draws [B, M] (z-space for regression, class idx (float) for
        classification) and, for classification, the predictive probs [B, M, C]."""
        if self.task == "regression":
            u = torch.rand(logits.shape[:-1], device=logits.device, generator=generator)
            u = u * (1 - 2 * self.u_clip) + self.u_clip  # truncate the half-normal tails of the bar distribution
            z = bar_icdf(self.bardist, logits, u)
            if self.y_range is not None:
                st_ = getattr(self, "_cur_state", None)
                if st_ is not None:
                    z = z.clamp(self._znorm(torch.tensor(self.y_range[0], device=z.device), st_), self._znorm(torch.tensor(self.y_range[1], device=z.device), st_))
            return z, None
        lg = logits[..., : self.n_classes]
        probs = torch.softmax(lg, -1)
        g = -torch.log(-torch.log(torch.rand(lg.shape, device=lg.device, generator=generator).clamp_(1e-20, 1)))
        hard = torch.argmax(lg + g, -1)
        if differentiable:
            soft = torch.softmax((lg + g) / 0.5, -1)
            onehot = F.one_hot(hard, self.n_classes).to(soft)
            st = onehot + soft - soft.detach()  # straight-through
            classes = torch.arange(self.n_classes, device=lg.device, dtype=st.dtype)
            return (st * classes).sum(-1), probs
        return hard.to(torch.float32), probs

    def _fantasy_draws(self, state, ctx_x, ctx_y, ctx_w, eval_x, n_draws, generator, differentiable, G=1):
        """Chunked autoregressive fantasy sampling.
        ctx_x [N, C] or [G, N, C]; ctx_y [N] (z-space / class idx) shared; ctx_w [N, G] or None.
        Returns draws [G, n_draws, M] (regression: raw-scale labels) or [G, n_draws, M, C] (probs)."""
        M = eval_x.shape[0]
        Btot = G * n_draws
        dev = self.device
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(_draw_seed(generator)))
        chunks = list(torch.chunk(perm.to(dev), self.n_chunks))
        cx = ctx_x.to(dev)
        if cx.dim() == 3:
            cx = cx.unsqueeze(1).expand(G, n_draws, -1, -1).reshape(Btot, cx.shape[1], cx.shape[2])
        cy = ctx_y.to(dev).float().unsqueeze(1).expand(-1, Btot)  # [N, Btot]
        cw = None
        if ctx_w is not None:
            cw = ctx_w.to(dev).float().unsqueeze(2).expand(-1, G, n_draws).reshape(-1, Btot)  # [N, Btot]
        fant = torch.zeros(Btot, M, device=dev)
        probs_all = torch.zeros(Btot, M, self.n_classes, device=dev) if self.task != "regression" else None
        ex = eval_x.to(dev)
        for idx in chunks:
            qx = ex[idx]
            logits = self._batched_forward(cx, cy, qx, cw, differentiable)
            z, probs = self._sample_from_logits(logits, generator, differentiable)  # [Btot, |idx|]
            fant = fant.index_copy(1, idx, z) if differentiable else fant.index_copy_(1, idx, z)
            if probs is not None:
                probs_all = probs_all.index_copy(1, idx, probs)
            # append fantasies to the context
            if cx.dim() == 3:
                cx = torch.cat([cx, qx.unsqueeze(0).expand(Btot, -1, -1)], 1)
            else:
                cx = torch.cat([cx, qx], 0)
            cy = torch.cat([cy, z.t()], 0)
            if cw is not None:
                cw = torch.cat([cw, torch.ones(z.shape[1], Btot, device=dev)], 0)
        if self.draw_type == "mean":
            logits = self._batched_forward(cx, cy, ex, cw, differentiable)
            if self.task == "regression":
                out = self._deznorm(self.bardist.mean(logits.float()), state)  # [Btot, M]
            else:
                out = torch.softmax(logits[..., : self.n_classes], -1)  # [Btot, M, C]
        else:
            out = self._deznorm(fant, state) if self.task == "regression" else probs_all
        return out.view(G, n_draws, *out.shape[1:])

    # ----- UQ interface -----
    def condition(self, train_x, train_y):
        X = train_x.to(self.device, torch.float32)
        y = train_y.to(self.device).float()
        if self.task == "regression":
            y_mean, y_std = y.mean(), y.std(correction=0).clamp_min(1e-6)
        else:
            y_mean, y_std = torch.zeros((), device=self.device), torch.ones((), device=self.device)
        tx = self._fit_tx(X)
        st = {"X": X, "y_raw": y, "y_mean": y_mean, "y_std": y_std, "y": (y - y_mean) / y_std if self.task == "regression" else y, "tx": tx}
        if tx is not None:
            st["X_raw"] = X; st["X"] = self._apply_tx(X, tx)
        return st

    def _fit_tx(self, X):
        if self.input_transform == "none":
            return None
        xd = X.double()
        if self.input_transform == "fix":
            sd = xd.std(0) if xd.shape[0] > 1 else torch.zeros(xd.shape[1], device=xd.device, dtype=xd.dtype)
            return {"kind": "fix", "shift": torch.where(sd < 1e-6, xd[0], torch.zeros_like(sd))}
        med = xd.median(0).values
        s = torch.quantile(xd, 0.75, dim=0) - torch.quantile(xd, 0.25, dim=0)
        half_range = (xd.max(0).values - xd.min(0).values) / 2.0
        s = torch.where(s > 1e-9, s, torch.where(half_range > 1e-9, half_range, torch.ones_like(s)))
        return {"kind": self.input_transform, "med": med, "scale": s}

    def _apply_tx(self, x, tx):
        if tx is None:
            return x
        xd = x.to(self.device).double()
        if tx["kind"] == "fix":
            return (xd - tx["shift"]).float()
        z = (xd - tx["med"]) / tx["scale"]
        if tx["kind"] == "asinh":
            return torch.asinh(z).float()
        B = 3.0 if tx["kind"] == "squash3" else 10.0
        return (z / torch.sqrt(1.0 + (z / B) ** 2)).float()

    def _tx(self, x, state):
        """features of evaluation / pool points in the units of state['X']"""
        return self._apply_tx(x, state.get("tx")) if state.get("tx") is not None else x

    def posterior_draws(self, state, eval_x, n_draws=None, generator=None):
        eval_x = self._tx(eval_x, state)
        n = int(n_draws or self.n_draws)
        self._cur_state = state
        with torch.no_grad():
            d = self._fantasy_draws(state, state["X"], state["y"], None, eval_x, n, generator, False, G=1)
        return d[0]  # [S, M(, C)]

    def predictive(self, state, xq):
        """One-pass marginal predictive at xq: (mean, std) for regression; probs [M, C] for classification."""
        xq = self._tx(xq, state)
        with torch.no_grad():
            logits = self._batched_forward(state["X"], state["y"].unsqueeze(1), xq.to(self.device), None, False)[0]
        if self.task == "regression":
            mean, var = self._moments(logits, state)
            return mean, var.clamp_min(0).sqrt()
        return torch.softmax(logits[..., : self.n_classes], -1)

    def posterior_mean_std(self, state, xq):
        if self.task == "regression":
            return self.predictive(state, xq)
        p = self.predictive(state, xq)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        return p, ent  # "uncertainty" = predictive entropy for classification

    def pointwise_std(self, state, pool_x):
        return self.posterior_mean_std(state, pool_x)[1]

    def _target_mean_from_logits(self, logits, state, target):
        """logits [B, M, n_out] -> E[Q] per batch element [B] from TabPFN's marginal predictive."""
        if self.task == "regression":
            m, v = self._moments(logits, state)
            return target.mean_from_moments(m, v.clamp_min(0))
        return target.mean_from_probs(torch.softmax(logits[..., : self.n_classes], -1))

    def target_mean(self, state, eval_x, target):
        eval_x = self._tx(eval_x, state)
        with torch.no_grad():
            logits = self._batched_forward(state["X"], state["y"].unsqueeze(1), eval_x.to(self.device), None, False)
        return self._target_mean_from_logits(logits, state, target)[0]

    def rollout_target_mean(self, state, eval_x, extra_x, extra_y, weights, target, differentiable=False):
        """E[Q | D, pseudo-update] for each rollout, one TabPFN pass (batch over rollouts): [G]."""
        eval_x, extra_x = self._tx(eval_x, state), self._tx(extra_x, state)
        G = weights.shape[0]
        ey, per = self._extra_y(extra_y, G, state)  # per: one imagined label vector per rollout ([G, N_extra])
        X, y = state["X"], state["y"]
        ex = eval_x.to(self.device)
        if not differentiable:
            sel = [(weights[g] > 0).nonzero().squeeze(1) for g in range(G)]
            b = sel[0].numel()
            assert all(s_.numel() == b for s_ in sel), "hard rollouts need equal batch sizes"
            ctx_x = torch.stack([torch.cat([X, extra_x.to(self.device)[s_]], 0) for s_ in sel], 0)
            ctx_y = torch.stack([torch.cat([y, (ey[g] if per else ey)[s_]], 0) for g, s_ in enumerate(sel)], 1)
            logits = self._batched_forward(ctx_x, ctx_y, ex, None, False)
            return self._target_mean_from_logits(logits, state, target)
        W = weights.to(self.device, torch.float32)
        if self.soft_topm is not None and self.soft_topm < W.shape[1]:
            top = torch.topk(W, self.soft_topm, dim=1).indices.unique()
            extra_x, ey, W = extra_x[top.cpu()], (ey[:, top] if per else ey[top]), W[:, top]
        ctx_x = torch.cat([X, extra_x.to(self.device)], 0)
        ctx_y = torch.cat([y.unsqueeze(1).expand(-1, G), ey.t()], 0) if per else torch.cat([y, ey], 0).unsqueeze(1).expand(-1, G)
        ctx_w = torch.cat([torch.ones(X.shape[0], G, device=self.device), W.t()], 0)
        logits = self._batched_forward(ctx_x, ctx_y, ex, ctx_w, True)
        return self._target_mean_from_logits(logits, state, target)

    def _extra_y(self, extra_y, G, state):
        """Imagined labels for a rollout batch: [N_extra] (shared) or [G, N_extra] (one vector per rollout).
        Returns (labels on the model scale, per_rollout flag)."""
        per = extra_y.dim() == 2 and extra_y.shape[0] == G
        ey = extra_y if (per or extra_y.dim() == 1) else extra_y[0]
        ey = ey.to(self.device).float()
        if self.task == "regression":
            ey = self._znorm(ey, state)
        return ey, per

    def sample_pool_labels(self, state, pool_x, n=1, generator=None):
        """Joint draw of imagined labels for the pool (chunked autoregressive), raw scale / class idx."""
        pool_x = self._tx(pool_x, state)
        saved = self.n_chunks, self.draw_type
        self.n_chunks, self.draw_type = self.n_pool_chunks, "labels"
        try:
            with torch.no_grad():
                d = self._fantasy_draws(state, state["X"], state["y"], None, pool_x, n, generator, False, G=1)[0]
        finally:
            self.n_chunks, self.draw_type = saved
        if self.task == "regression":
            return d  # [n, N_pool]
        # for classification, _fantasy_draws returns probs; sample labels from them
        return torch.distributions.Categorical(probs=d).sample().long()

    def rollout_draws(self, state, eval_x, extra_x, extra_y, weights, n_draws=None, differentiable=False, generator=None):
        """Draws after the (weighted) pseudo-update.  weights [G, N_extra]; extra_y [N_extra] shared.
        Returns [G, S, M(, C)]."""
        eval_x, extra_x = self._tx(eval_x, state), self._tx(extra_x, state)
        S = int(n_draws or self.n_draws)
        G = weights.shape[0]
        ey, per = self._extra_y(extra_y, G, state)
        X, y = state["X"], state["y"]
        if not differentiable:
            # hard batches: per-rollout contexts of equal size (train + selected items)
            sel = [(weights[g] > 0).nonzero().squeeze(1) for g in range(G)]
            b = sel[0].numel()
            assert all(s.numel() == b for s in sel), "hard rollouts need equal batch sizes"
            ctx_x = torch.stack([torch.cat([X, extra_x.to(self.device)[s]], 0) for s in sel], 0)  # [G, N+b, C]
            ctx_y = torch.stack([torch.cat([y, (ey[g] if per else ey)[s]], 0) for g, s in enumerate(sel)], 1)  # [N+b, G]
            with torch.no_grad():
                return self._fantasy_draws_multi(state, ctx_x, ctx_y, None, eval_x, S, generator, False, G)
        W = weights.to(self.device, torch.float32)  # [G, N_extra]
        if self.soft_topm is not None and self.soft_topm < W.shape[1]:
            # keep only the top-m pool items (by weight) per rollout in the soft context; others are
            # exactly excluded.  Items are the union over rollouts so the context rows are shared.
            top = torch.topk(W, self.soft_topm, dim=1).indices.unique()
            extra_x, ey, W = extra_x[top.cpu()], (ey[:, top] if per else ey[top]), W[:, top]
        ctx_x = torch.cat([X, extra_x.to(self.device)], 0)
        ctx_y = torch.cat([y.unsqueeze(1).expand(-1, G), ey.t()], 0) if per else torch.cat([y, ey], 0)
        ctx_w = torch.cat([torch.ones(X.shape[0], G, device=self.device), W.t()], 0)  # [N, G]
        return self._fantasy_draws(state, ctx_x, ctx_y, ctx_w, eval_x, S, generator, differentiable, G)

    def _fantasy_draws_multi(self, state, ctx_x, ctx_y, ctx_w, eval_x, n_draws, generator, differentiable, G):
        """Variant of _fantasy_draws with per-rollout labels ctx_y [N, G] (hard batches)."""
        M = eval_x.shape[0]
        Btot = G * n_draws
        dev = self.device
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(_draw_seed(generator)))
        chunks = list(torch.chunk(perm.to(dev), self.n_chunks))
        cx = ctx_x.to(dev).unsqueeze(1).expand(G, n_draws, -1, -1).reshape(Btot, ctx_x.shape[1], ctx_x.shape[2])
        cy = ctx_y.to(dev).float().unsqueeze(2).expand(-1, G, n_draws).reshape(-1, Btot)
        fant = torch.zeros(Btot, M, device=dev)
        probs_all = torch.zeros(Btot, M, self.n_classes, device=dev) if self.task != "regression" else None
        ex = eval_x.to(dev)
        for idx in chunks:
            qx = ex[idx]
            logits = self._batched_forward(cx, cy, qx, None, False)
            z, probs = self._sample_from_logits(logits, generator, False)
            fant.index_copy_(1, idx, z)
            if probs is not None:
                probs_all.index_copy_(1, idx, probs)
            cx = torch.cat([cx, qx.unsqueeze(0).expand(Btot, -1, -1)], 1)
            cy = torch.cat([cy, z.t()], 0)
        if self.draw_type == "mean":
            logits = self._batched_forward(cx, cy, ex, None, False)
            out = self._deznorm(self.bardist.mean(logits.float()), state) if self.task == "regression" else torch.softmax(logits[..., : self.n_classes], -1)
        else:
            out = self._deznorm(fant, state) if self.task == "regression" else probs_all
        return out.view(G, n_draws, *out.shape[1:])
