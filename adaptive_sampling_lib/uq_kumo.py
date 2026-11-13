"""Kumo-Tabular (NVIDIA, structured-data-models) as an uncertainty-quantification module for the planner (regression and
classification).

Belief.  Kumo-Tabular's in-context predictive given the labelled data D: 999 quantiles q_k at levels k/1000 for every
query point.  Moments come from the quantile grid (full: all 999 levels; range-conditioned: the levels inside the task's
label range, as TabPFNModule's quantile-grid truncation).  Joint posterior draws over the evaluation set are produced by
chunked autoregressive fantasy sampling as in TabPFNModule: the evaluation points are visited in K random chunks, the
labels of chunk k are drawn from the predictive given D plus the fantasised chunks < k (inverse CDF by linear
interpolation of the quantile grid) and appended to the context.

Classification.  The classifier is a 10-class network; for C <= 10 classes Kumo uses its first C logits and a softmax (its
error-correcting codes apply only above 10 classes).  Labels are class indices (no target standardisation).  As for the
TabPFN classifier (draw_type 'mean'), a joint posterior draw samples the evaluation labels chunk by chunk and then returns
the predictive class probabilities of all evaluation points given D plus all fantasised labels; imagined pool labels are
the sampled labels themselves.  The uncertainty baselines score the predictive entropy, as for TabPFN.

Preprocessing.  One ensemble member of Kumo's own recipe, without its random column shuffle and sign flips (the model is
trained to be invariant to both): drop the columns that are constant on D, standardise every feature with the context
mean and population standard deviation (+1e-6), clip to [-100, 100], fit the 4-sigma soft clip (ClipSigma) on the
context, and standardise the target with the context mean and population standard deviation.  The kept columns are
fixed by D (condition) for the whole epoch, so the planner's candidate batches cannot change them; every other statistic
is recomputed for each context.

Weighted (fractional) pseudo-update.  The planner's relaxed batch assigns a weight a_i in [0,1] to every pool item.  The
three attentions whose keys are context rows (the inducing points of every column block attending to the context rows,
and the context-to-context and query-to-context attention of every ICL layer) get the additive attention-logit bias
log a_i, their logarithmic query scaling uses the weighted context size sum_i a_i instead of the row count, and every
preprocessing statistic above is a-weighted (the soft clip's inlier mask is piecewise constant in a).  Binary weights
reproduce the ordinary hard update exactly; the model's weights are frozen.

Preprocessing in the planner's pseudo-updates (fixed_stats).  With every statistic a-weighted ('none'), the pathwise
planner exploits the feature standardisation: on Bike, whose integer-coded features are nearly constant within the training
cluster, weight on near-duplicate training-cluster items shrinks those columns' weighted spread, which rescales every
out-of-cluster input and moves the estimate of Q, so the planner collapsed onto the training cluster (92-98 % of its picks
at 10 steps against 22-38 % for REINFORCE with baseline).  With fixed_stats='both' the pseudo-updates (rollout_target_mean,
hard and soft alike) use D's feature and target statistics for the whole epoch, as the TabPFN module's input transform is
fitted on D and fixed within an epoch; the belief after a real update (condition) and the tracked and imagined draws keep
the context's own statistics, as in Kumo's recipe.  Fixing the target statistics alone does not remove the collapse.

Self-attention of a soft row (self_attention).  With 'weighted' (the first Kumo grids) a row of weight a also attends to
itself with weight a in the ICL layers' context-to-context attention, so at small a its own representation is built almost
entirely from other rows; evaluation points near it still attend to it strongly, and the update saturates: on California a
pool item at weight 0.01 moved the estimate of Q more than the same item at weight 1, the relaxed objective was 60-70 % drift,
and the pathwise planner, optimised for more steps, parked weight on inert training-cluster items whose top-b rounding had
almost no gain (the weight-curve diagnostics are kept with the reports, outside the repository; the test of the
relaxations against each other is experiments/kumo/regression/stage1.py).  With 'full' every row
attends to itself at full weight while counting with weight a for the other rows; the single-item curve then rises smoothly
from 0 to its hard value, as with TabPFN, and binary weights reproduce the hard update as before.

Slot relaxations (relax).  'weighted' (default) is the fractional update above.  'avg_input' and 'avg_emb' instead relax
the batch as b slot rows, each a convex combination of pool items given by the b successive relaxed one-hot vectors of the
planner's soft top-k (subset.soft_topk_slots): 'avg_input' averages the items' raw features and imagined labels, so a
one-hot slot is exactly the item and the update is the hard one; 'avg_emb' averages the items' row embeddings (computed
once, with D's rows as the only context of the column blocks) and labels before the ICL stage, so a one-hot slot differs
from the hard update only in that the batch's rows do not enter the column blocks' context.  Classification labels are
averaged as class probabilities, whose mixture of label-embedding rows replaces the lookup (_KCTX['soft_y']).

Objectives.  rollout_target_mean gives E[Q] after the pseudo-update (the 'gain' and 'gain_paired' objectives);
rollout_draws gives joint draws over the evaluation points after it (the 'risk' objective, the expected posterior variance
of Q), differentiable through reparameterised draws (inverse-CDF interpolation of the quantile grid; straight-through
Gumbel one-hot labels entering the context as soft labels), with activation checkpointing; not for relax='avg_emb'.

Precision.  Everything the planner differentiates or compares (E[Q] for the gain objective, the rollouts, the uncertainty
scores) runs in float32.  The sampling-only paths (the tracked joint draws and the imagined pool labels) run under float16
autocast (sample_dtype), 3.4x faster; their quantiles differ from float32 by 3e-4 on average for a 0.1-99.9 % width of 1.7.
Triton (used by the package under autocast) needs a node-local cache: a cache shared over nodes holds binaries built
against another glibc.
"""
from __future__ import annotations

import os
import socket
import sys

from adaptive_sampling_lib.paths import MODEL_CACHE  # noqa: E402

os.environ.setdefault("HF_HUB_CACHE", os.path.join(MODEL_CACHE, "hf_hub"))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_{os.environ.get('USER', 'user')}_{socket.gethostname()}")
SDM_PATH = os.environ.get("SDM_PATH", os.path.join(MODEL_CACHE, "sdm_pkgs"))
if SDM_PATH not in sys.path:
    sys.path.insert(0, SDM_PATH)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sdm.models import KumoTabular  # noqa: E402
from sdm.nn.attention import SDPA  # noqa: E402

from .uq_tabpfn import _draw_seed  # noqa: E402

# --------------------------------------------------------------------------------------
# Patch for the weighted context
# --------------------------------------------------------------------------------------
_KCTX = {"logw": None, "sumw": None, "self_full": False, "soft_y": None}  # [B, N] log weights of the context rows and [B] their sum,
# during a weighted forward; self_full: in the ICL layers' context-to-context attention every row attends to itself at full
# weight; soft_y: [B, N, C] class probabilities of the context rows replacing the classifier's label lookup (soft labels)
_ORIG_SDPA_FORWARD = SDPA.forward


def _weighted_sdpa_forward(self, query, key, value, seqused_key_value=None, attn_mask=None):
    logw = _KCTX["logw"]
    if (not getattr(self, "_kumo_ctx", False) or logw is None or key.size(-3) != logw.shape[1]
            or attn_mask is not None or seqused_key_value is not None):
        return _ORIG_SDPA_FORWARD(self, query, key, value, seqused_key_value=seqused_key_value, attn_mask=attn_mask)
    if query.numel() == 0:
        return query
    batch_shape = torch.broadcast_shapes(query.size()[:-3], key.size()[:-3], value.size()[:-3])
    B, N = logw.shape
    assert batch_shape[0] == B, (batch_shape, logw.shape)
    nb = len(batch_shape)
    if self.query_scaling is not None:  # log(n) query scaling with the weighted number of context rows
        query = self.query_scaling(query, key_len=_KCTX["sumw"].view(B, *([1] * (nb - 1)), 1))
    qs, ks, vs = query.size()[-3:], key.size()[-3:], value.size()[-3:]
    Q = qs[0]
    query = query.expand(batch_shape + qs).reshape(-1, *qs)
    key = key.expand(batch_shape + ks).reshape(-1, *ks)
    value = value.expand(batch_shape + vs).reshape(-1, *vs)
    bias = logw.to(query.dtype).view(B, *([1] * (nb - 1)), 1, N).expand(*batch_shape, Q, N).reshape(-1, 1, Q, N)
    if _KCTX["self_full"] and getattr(self, "_kumo_icl", False) and Q == N:  # context rows attending to context rows
        bias = (bias * (1 - torch.eye(N, device=bias.device, dtype=bias.dtype))).contiguous()
    out = F.scaled_dot_product_attention(query.transpose(-3, -2), key.transpose(-3, -2), value.transpose(-3, -2), attn_mask=bias,
                                         enable_gqa=query.size(-2) != key.size(-2), scale=self.scale).transpose(-3, -2)
    return out.reshape(batch_shape + out.size()[-3:])


SDPA.forward = _weighted_sdpa_forward


def _soft_label_lookup(emb):
    """Let a label-embedding table take soft labels: while _KCTX['soft_y'] holds class probabilities whose leading shape
    matches the index tensor, return their mixture of the table's rows instead of the lookup."""
    orig = emb.forward

    def forward(idx):
        sy = _KCTX["soft_y"]
        if sy is not None and tuple(idx.shape) == tuple(sy.shape[:-1]):
            return sy.to(emb.weight.dtype) @ emb.weight[: sy.shape[-1]]
        return orig(idx)
    emb.forward = forward

U_LEVELS = torch.arange(1, 1000, dtype=torch.float64) / 1000.0  # levels of Kumo's 999 regression quantiles


class KumoModule:
    name = "kumo"

    def __init__(self, size: str = "medium", device="cuda", n_draws: int = 32, n_chunks: int = 4, n_pool_chunks: int = 5,
                 soft_topm: int | None = None, y_range: tuple | None = None, u_clip: float = 0.005, max_batch_rows: int = 60_000,
                 checkpoint: bool = False, sample_dtype: torch.dtype | None = torch.float16, fixed_stats: str = "none",
                 task: str = "regression", n_classes: int = 2, self_attention: str = "weighted", relax: str = "weighted"):
        assert relax in ("weighted", "avg_input", "avg_emb"), relax
        self.relax = relax
        self.slot_relaxation = relax != "weighted"  # the planner then passes b relaxed one-hot slots [G, b, n] (soft_topk_slots)
        assert self_attention in ("weighted", "full"), self_attention
        self.self_attention = self_attention  # 'full': a soft row attends to itself at full weight in the ICL layers (see the docstring)
        assert task in ("regression", "classification"), task
        assert task == "regression" or (2 <= n_classes <= 10 and y_range is None), (n_classes, y_range)
        self.n_classes = int(n_classes)
        assert fixed_stats in ("none", "target", "features", "both"), fixed_stats
        self.fixed_stats = fixed_stats  # preprocessing statistics taken from D for the whole epoch instead of each context's
        self.sample_dtype = sample_dtype
        self.task, self.device = task, torch.device(device)
        self.n_draws, self.n_chunks, self.n_pool_chunks = int(n_draws), int(n_chunks), int(n_pool_chunks)
        self.soft_topm, self.u_clip, self.max_batch_rows, self.checkpoint = soft_topm, float(u_clip), int(max_batch_rows), checkpoint
        self.grad_batch = None  # batch elements per checkpointed segment (rollout_draws sets 8 for its differentiable draws)
        self.y_range = None if y_range is None else (float(y_range[0]), float(y_range[1]))
        self.draw_type = "labels" if task == "regression" else "mean"
        wrapper = KumoTabular(task=task, size=size, device=self.device)
        self.net = wrapper.models[task].eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        n_tag = 0
        for n, mod in self.net.named_modules():  # the attentions whose keys are context rows
            if isinstance(mod, SDPA) and (".inducing_block." in n or n.startswith("icl_block.layers.")):
                mod._kumo_ctx = True; mod._kumo_icl = n.startswith("icl_block."); n_tag += 1
        assert n_tag == len(self.net.row_embedding.col_blocks) + len(self.net.icl_block.layers), n_tag
        if task == "classification":
            for emb in (self.net.row_embedding.y_emb, self.net.icl_block.y_emb):
                _soft_label_lookup(emb)
        self.u = U_LEVELS.to(self.device)

    # ----- raw forward -----
    @staticmethod
    def _stats(cx, cy, w, keep):
        """Preprocessing statistics of a (weighted) context: cx [B, N, C], cy [B, N] (raw labels), w [B, N] ->
        feature standardisation (mean, scale), soft-clip centre and spread (mu, sd), target mean and scale (ym, ys)."""
        x = cx.double()[..., keep]
        w = w.double(); sw = w.sum(1, keepdim=True); wn = (w / sw).unsqueeze(-1)  # [B, N, 1]
        mean = (wn * x).sum(1, keepdim=True)
        scale = ((wn * (x - mean) ** 2).sum(1, keepdim=True)).sqrt() + 1e-6  # Standardize(eps=1e-6)
        xc = ((x - mean) / scale).clamp(-100.0, 100.0)
        # ClipSigma(threshold=4): bounds from the context (sample variance), refitted on the context's inliers
        ww = w.unsqueeze(-1)
        m1 = (wn * xc).sum(1, keepdim=True)
        s1 = ((ww * (xc - m1) ** 2).sum(1, keepdim=True) / (sw.unsqueeze(-1) - 1).clamp_min(1)).sqrt().clamp_min(1e-6)
        inl = ((xc >= m1 - 4 * s1) & (xc <= m1 + 4 * s1)).double() * ww
        cnt = inl.sum(1, keepdim=True)
        km = (inl * xc).sum(1, keepdim=True) / cnt.clamp_min(1e-12)
        ks = ((inl * (xc - km) ** 2).sum(1, keepdim=True) / (cnt - 1).clamp_min(1)).sqrt().clamp_min(1e-6)
        mu, sd = torch.where(cnt > 0, km, m1), torch.where(cnt > 0, ks, s1)
        # target: Standardize (population variance; unit scale if constant)
        y = cy.double()
        ym = (w * y).sum(1, keepdim=True) / sw
        yv = (w * (y - ym) ** 2).sum(1, keepdim=True) / sw
        const = yv <= sw * torch.finfo(yv.dtype).eps * yv + (sw * ym * torch.finfo(yv.dtype).eps) ** 2
        ys = torch.where(const, torch.ones_like(yv), yv.clamp_min(1e-300).sqrt())
        return {"mean": mean, "scale": scale, "mu": mu, "sd": sd, "ym": ym, "ys": ys}

    def _prep(self, cx, cy, qx, w, keep, fixed=None):
        """cx [B, N, C], cy [B, N] (raw labels / class indices), qx [B, M, C], w [B, N] -> network inputs and the target
        scale.  The statistics are those of the weighted context, except the ones that self.fixed_stats takes from `fixed`
        (D's).  Classification: the labels are passed as class indices and the target scale is unused."""
        s = self._stats(cx, cy, w, keep)
        if fixed is not None and self.fixed_stats in ("features", "both"):
            s.update({k: fixed[k] for k in ("mean", "scale", "mu", "sd")})
        if fixed is not None and self.fixed_stats in ("target", "both"):
            s.update({k: fixed[k] for k in ("ym", "ys")})
        x = ((torch.cat([cx, qx], 1).double()[..., keep] - s["mean"]) / s["scale"]).clamp(-100.0, 100.0)
        la = x.abs().log1p()
        x = torch.minimum(la + (s["mu"] + 4 * s["sd"]), torch.maximum((s["mu"] - 4 * s["sd"]) - la, x))
        if self.task == "classification":
            return x.float(), cy.round().long(), s["ym"], s["ys"]
        return x.float(), ((cy.double() - s["ym"]) / s["ys"]).float(), s["ym"], s["ys"]

    def forward_raw(self, ctx_x, ctx_y, query_x, keep, ctx_w=None, grad=False, stats=None, soft_y=None):
        """ctx_x [N, C] or [B, N, C]; ctx_y [N] or [N, B] (raw labels); query_x [M, C] or [B, M, C]; ctx_w [N, B] or None;
        stats: D's preprocessing statistics (used for the parts named by self.fixed_stats); soft_y: [B, N, C] class
        probabilities of the context rows (classification soft labels; ctx_y then only fixes the shapes).
        Returns the 999 predictive quantiles [B, M, 999] in raw label units (sorted); classification: the predictive class
        probabilities [B, M, n_classes] (float64)."""
        if ctx_y.dim() == 1:
            ctx_y = ctx_y.unsqueeze(1)
        N, B = ctx_y.shape
        cx = (ctx_x if ctx_x.dim() == 3 else ctx_x.unsqueeze(0).expand(B, -1, -1)).to(self.device)
        qx = (query_x if query_x.dim() == 3 else query_x.unsqueeze(0).expand(B, -1, -1)).to(self.device)
        cy = ctx_y.t().to(self.device)
        w = torch.ones(B, N, device=self.device) if ctx_w is None else ctx_w.t().to(self.device, torch.float32)
        assert not (torch.isnan(cx).any() or torch.isnan(qx).any()), "missing values are not supported"

        def _run(cx_, cy_, qx_, w_, sy_=None):
            x, y, ym, ys = self._prep(cx_, cy_, qx_, w_, keep, stats)
            weighted = ctx_w is not None
            _KCTX["logw"] = torch.log(w_.clamp_min(1e-30)) if weighted else None
            _KCTX["sumw"] = w_.sum(1) if weighted else None
            _KCTX["self_full"] = weighted and self.self_attention == "full"
            _KCTX["soft_y"] = sy_
            try:
                out = self.net(x, y, torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device))  # [B, M, 999 or 10]
            finally:
                _KCTX["logw"] = _KCTX["sumw"] = None; _KCTX["self_full"] = False; _KCTX["soft_y"] = None
            if self.task == "classification":
                return torch.softmax(out[..., : self.n_classes].double(), -1)
            return out.double().sort(-1).values * ys.unsqueeze(-1) + ym.unsqueeze(-1)

        sy = None if soft_y is None else soft_y.to(self.device, torch.float32)
        if grad:
            with torch.enable_grad():
                if self.checkpoint:  # one checkpointed segment per group of grad_batch batch elements (peak memory of one group)
                    gb = self.grad_batch or B
                    return torch.cat([torch.utils.checkpoint.checkpoint(_run, cx[s:s + gb], cy[s:s + gb], qx[s:s + gb], w[s:s + gb],
                                                                        None if sy is None else sy[s:s + gb], use_reentrant=False)
                                      for s in range(0, B, gb)], 0)
                return _run(cx, cy, qx, w, sy)
        with torch.no_grad():
            per = max(1, self.max_batch_rows // (N + qx.shape[1]))
            return torch.cat([_run(cx[s:s + per], cy[s:s + per], qx[s:s + per], w[s:s + per], None if sy is None else sy[s:s + per])
                              for s in range(0, B, per)], 0)

    # ----- quantile helpers -----
    def _moments(self, q):
        """q [..., 999] -> predictive mean and variance: of the whole grid, or of the levels inside y_range."""
        if self.y_range is None:
            m = q.mean(-1)
            return m, ((q - m.unsqueeze(-1)) ** 2).mean(-1)
        lo, hi = self.y_range
        inside = ((q >= lo) & (q <= hi)).to(q)
        n_in = inside.sum(-1, keepdim=True)
        w = torch.where(n_in > 0, inside, torch.ones_like(inside))
        qq = torch.where(n_in > 0, q, q.clamp(lo, hi))  # fallback (all mass outside): clamped values
        n = w.sum(-1)
        m = (qq * w).sum(-1) / n
        return m, (((qq - m.unsqueeze(-1)) ** 2) * w).sum(-1) / n

    def _sample(self, q, generator):
        """Inverse-CDF draws from the quantile grid q [B, M, 999] at u ~ U(u_clip, 1 - u_clip) (linear interpolation);
        classification: class indices (as float) drawn from the class probabilities q [B, M, n_classes]."""
        if self.task == "classification":
            u = torch.rand(q.shape[:-1], device=q.device, generator=generator, dtype=torch.float64)
            return (u.unsqueeze(-1) > q.cumsum(-1)).sum(-1).clamp_max(self.n_classes - 1).double()
        u = torch.rand(q.shape[:-1], device=q.device, generator=generator, dtype=torch.float64) * (1 - 2 * self.u_clip) + self.u_clip
        pos = (u * 1000.0 - 1.0).clamp(0.0, 998.0)  # fractional index into levels 1/1000 .. 999/1000
        i0 = pos.floor().long().clamp(0, 997)
        fr = (pos - i0.double()).unsqueeze(-1)
        q0, q1 = q.gather(-1, i0.unsqueeze(-1)), q.gather(-1, (i0 + 1).unsqueeze(-1))
        z = (q0 + fr * (q1 - q0)).squeeze(-1)
        if self.y_range is not None:
            z = z.clamp(*self.y_range)
        return z

    def _fantasy_draws(self, state, ctx_x, ctx_y, eval_x, n_draws, generator, out="labels"):
        """Chunked autoregressive label draws over eval_x given the (hard) context; ctx_y [N] or [N, G] with ctx_x [N, C]
        or [G, N, C].  Returns [G, n_draws, M] raw labels (class indices), or with out='probs' (classification) the
        predictive class probabilities [G, n_draws, M, n_classes] given the context plus all fantasised labels."""
        G = 1 if ctx_y.dim() == 1 else ctx_y.shape[1]
        Btot, M, dev = G * n_draws, eval_x.shape[0], self.device
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(_draw_seed(generator)))
        cx = ctx_x.to(dev)
        cx = (cx.unsqueeze(0) if cx.dim() == 2 else cx).unsqueeze(1).expand(G, n_draws, -1, -1).reshape(Btot, cx.shape[-2], cx.shape[-1])
        cy = ctx_y.to(dev).double().reshape(ctx_y.shape[0], G).unsqueeze(2).expand(-1, G, n_draws).reshape(-1, Btot)  # [N, Btot]
        fant = torch.zeros(Btot, M, device=dev, dtype=torch.float64)
        ex = eval_x.to(dev)
        for idx in torch.chunk(perm.to(dev), self.n_chunks):
            qx = ex[idx]
            with torch.autocast(dev.type, dtype=self.sample_dtype or torch.float16, enabled=self.sample_dtype is not None):
                qq = self.forward_raw(cx, cy, qx, state["keep"])
            z = self._sample(qq, generator)  # [Btot, |idx|]
            fant.index_copy_(1, idx, z)
            cx = torch.cat([cx, qx.unsqueeze(0).expand(Btot, -1, -1)], 1)
            cy = torch.cat([cy, z.t()], 0)
        if out == "probs":
            with torch.autocast(dev.type, dtype=self.sample_dtype or torch.float16, enabled=self.sample_dtype is not None):
                pr = self.forward_raw(cx, cy, ex, state["keep"])  # [Btot, M, n_classes]
            return pr.float().view(G, n_draws, M, -1)
        return fant.float().view(G, n_draws, M)

    # ----- UQ interface -----
    def condition(self, train_x, train_y):
        X = train_x.to(self.device, torch.float32)
        y = train_y.to(self.device).float()
        keep = (X != X[:1]).any(0) if X.shape[0] > 1 else torch.ones(X.shape[1], dtype=torch.bool, device=self.device)  # DropConstantColumns on D
        stats = self._stats(X.unsqueeze(0), y.unsqueeze(0), torch.ones(1, X.shape[0], device=self.device), keep)  # D's preprocessing statistics
        return {"X": X, "y": y, "keep": keep, "stats": stats}

    def predictive(self, state, xq):
        """(mean, std) for regression; class probabilities [M, C] for classification."""
        q = self.forward_raw(state["X"], state["y"], xq.to(self.device), state["keep"])[0]
        if self.task == "classification":
            return q.float()
        m, v = self._moments(q)
        return m.float(), v.clamp_min(0).sqrt().float()

    def posterior_mean_std(self, state, xq):
        if self.task == "classification":
            p = self.predictive(state, xq)
            return p, -(p * torch.log(p.clamp_min(1e-12))).sum(-1)  # "uncertainty" = predictive entropy, as for TabPFN
        return self.predictive(state, xq)

    def pointwise_std(self, state, pool_x):
        return self.posterior_mean_std(state, pool_x)[1]

    def posterior_draws(self, state, eval_x, n_draws=None, generator=None):
        with torch.no_grad():
            return self._fantasy_draws(state, state["X"], state["y"], eval_x, int(n_draws or self.n_draws), generator,
                                       out="probs" if self.task == "classification" else "labels")[0]  # [S, M] or [S, M, C]

    def _target_mean(self, q, target):
        if self.task == "classification":
            return target.mean_from_probs(q.float())
        m, v = self._moments(q)
        return target.mean_from_moments(m.float(), v.clamp_min(0).float())

    def target_mean(self, state, eval_x, target):
        return self._target_mean(self.forward_raw(state["X"], state["y"], eval_x.to(self.device), state["keep"]), target)[0]

    def rollout_target_mean(self, state, eval_x, extra_x, extra_y, weights, target, differentiable=False):
        """E[Q | D, pseudo-update] for each rollout (one pass, batched over rollouts): [G]."""
        G = weights.shape[0]
        per = extra_y.dim() == 2 and extra_y.shape[0] == G
        ey = (extra_y if (per or extra_y.dim() == 1) else extra_y[0]).to(self.device).float()
        X, y, ex = state["X"], state["y"], eval_x.to(self.device)
        if weights.dim() == 3:  # slot relaxations: [G, b, n] relaxed one-hot slots
            return self._target_mean(self._slot_forward(state, ex, extra_x, ey, per, weights, differentiable), target)
        if not differentiable:
            sel = [(weights[g] > 0).nonzero().squeeze(1) for g in range(G)]
            assert all(s_.numel() == sel[0].numel() for s_ in sel), "hard rollouts need equal batch sizes"
            ctx_x = torch.stack([torch.cat([X, extra_x.to(self.device)[s_]], 0) for s_ in sel], 0)
            ctx_y = torch.stack([torch.cat([y, (ey[g] if per else ey)[s_]], 0) for g, s_ in enumerate(sel)], 1)
            return self._target_mean(self.forward_raw(ctx_x, ctx_y, ex, state["keep"], stats=state["stats"]), target)
        W = weights.to(self.device, torch.float32)
        if self.soft_topm is not None and self.soft_topm < W.shape[1]:
            top = torch.topk(W, self.soft_topm, dim=1).indices.unique()
            extra_x, ey, W = extra_x[top.cpu()] if extra_x.device.type == "cpu" else extra_x[top], (ey[:, top] if per else ey[top]), W[:, top]
        ctx_x = torch.cat([X, extra_x.to(self.device)], 0)
        ctx_y = torch.cat([y.unsqueeze(1).expand(-1, G), ey.t()], 0) if per else torch.cat([y, ey], 0).unsqueeze(1).expand(-1, G)
        ctx_w = torch.cat([torch.ones(X.shape[0], G, device=self.device), W.t()], 0)
        return self._target_mean(self.forward_raw(ctx_x, ctx_y, ex, state["keep"], ctx_w, grad=True, stats=state["stats"]), target)

    # ----- slot relaxations -----
    def _slot_context(self, state, extra_x, ey, per, P):
        """Context of relax='avg_input': D's rows plus b slot rows, each the P-average of the pool items' raw features and
        imagined labels.  P [G, b, n] (rows sum to 1); ey [n] or [G, n].  Returns ctx_x [G, N, C], ctx_y [N, G] and, for
        classification, the soft labels [G, N, C] (ctx_y then holds placeholder indices)."""
        G = P.shape[0]
        X, y = state["X"], state["y"]
        cand = extra_x.to(self.device, torch.float32)
        ctx_x = torch.cat([X.unsqueeze(0).expand(G, -1, -1), P @ cand], 1)
        if self.task == "classification":
            oh = F.one_hot(ey.long(), self.n_classes).float()  # [n, C] or [G, n, C]
            soft = torch.cat([F.one_hot(y.long(), self.n_classes).float().unsqueeze(0).expand(G, -1, -1), P @ oh], 1)
            return ctx_x, soft.argmax(-1).t().float(), soft
        ys = (P * (ey.unsqueeze(1) if per else ey)).sum(-1)  # [G, b]
        return ctx_x, torch.cat([y.unsqueeze(1).expand(-1, G), ys.t()], 0), None

    def _top_slots(self, extra_x, ey, per, P):
        """Restrict the slots to the soft_topm pool items with the largest total weight (renormalised over them)."""
        P = P.to(self.device, torch.float32)
        if self.soft_topm is None or self.soft_topm >= P.shape[-1]:
            return extra_x, ey, P
        top = torch.topk(P.sum(1), self.soft_topm, dim=1).indices.unique()
        extra_x = extra_x[top.cpu()] if extra_x.device.type == "cpu" else extra_x[top]
        P = P[..., top]
        return extra_x, (ey[:, top] if per else ey[top]), P / P.sum(-1, keepdim=True).clamp_min(1e-12)

    def _slot_forward(self, state, ex, extra_x, ey, per, P, grad):
        """Predictive at ex after adding b slot rows (relax='avg_input' or 'avg_emb'); P [G, b, n]."""
        extra_x, ey, P = self._top_slots(extra_x, ey, per, P)
        if self.relax == "avg_emb":
            assert not per, "relax='avg_emb' needs one imagined label vector shared by the rollouts"
            return self._forward_slots_emb(state, extra_x.to(self.device, torch.float32), ey, P, ex, grad)
        ctx_x, ctx_y, soft = self._slot_context(state, extra_x, ey, per, P)
        return self.forward_raw(ctx_x, ctx_y, ex, state["keep"], grad=grad, stats=state["stats"], soft_y=soft)

    def _row_embed_colctx(self, x, y, n_col):
        """Kumo's row-embedding stage (its gradient path) with labels on the first y.shape[-1] rows but only the first n_col
        rows as the column blocks' context.  x [B, R, C] preprocessed, y [B, R_lab] -> [B, R, K * D]."""
        re = self.net.row_embedding
        B_, R, C = x.shape
        R_lab, K, D = y.shape[-1], re.readout_token.size(-2), re.channels
        h = re.cell_embedding(x=x, categorical_mask=torch.zeros(B_, C, dtype=torch.bool, device=x.device), train_size=n_col)  # [B, R, C, D]
        ye = re.y_emb(y) if re.y_emb is not None else re.y_lin(y.unsqueeze(-1))  # [B, R_lab, D]
        h = torch.cat([h[:, :R_lab] + ye.unsqueeze(-2).to(h.dtype), h[:, R_lab:]], 1)
        for i, (cb, rb) in enumerate(zip(re.col_blocks, re.row_blocks)):
            if i > 0:
                rt, h = h.split([K, h.size(-2) - K], dim=-2)
            else:
                rt = re.readout_token.view(1, 1, K, D).expand(B_, R, K, D)
            h = h.transpose(-2, -3)  # [B, C, R, D]
            h = cb(query=h, key_value=h[..., :n_col, :])
            h = torch.cat([rt.to(h.dtype), h.transpose(-2, -3)], dim=-2)
            h = rb(query=h[..., :K, :] if i == len(re.row_blocks) - 1 else h, key_value=h)
        return re.norm(h).flatten(-2)

    def _forward_slots_emb(self, state, cand_x, cand_y, P, ex, grad):
        """relax='avg_emb': row embeddings of D's rows, the candidate pool items (with their imagined labels) and the evaluation
        points from one pass of the row-embedding stage whose column blocks see only D's rows; the b slot tokens are
        P-averages of the candidates' tokens and the ICL stage sees D, the slots (labels P-averaged) and the evaluation points."""
        X, y, keep = state["X"], state["y"], state["keep"]
        G, b, nc = P.shape
        nD = X.shape[0]
        with (torch.enable_grad() if grad else torch.no_grad()):
            x, yn, ym, ys = self._prep(torch.cat([X, cand_x], 0).unsqueeze(0), torch.cat([y, cand_y.float()], 0).unsqueeze(0), ex.unsqueeze(0),
                                       torch.ones(1, nD + nc, device=self.device), keep, state["stats"])
            tok = self.net.row_project(self._row_embed_colctx(x, yn, nD))[0]  # [nD + nc + M, D_icl]
            xi = torch.cat([tok[:nD].unsqueeze(0).expand(G, -1, -1), P @ tok[nD: nD + nc], tok[nD + nc:].unsqueeze(0).expand(G, -1, -1)], 1)
            if self.task == "classification":
                oh = F.one_hot(yn[0], self.n_classes).float()
                soft = torch.cat([oh[:nD].unsqueeze(0).expand(G, -1, -1), P @ oh[nD:]], 1)  # [G, nD + b, C]
                _KCTX["soft_y"] = soft
                try:
                    out = self.net.icl_block(x=xi, y=soft.argmax(-1))
                finally:
                    _KCTX["soft_y"] = None
                return torch.softmax(out[..., : self.n_classes].double(), -1)
            out = self.net.icl_block(x=xi, y=torch.cat([yn[0, :nD].unsqueeze(0).expand(G, -1), P @ yn[0, nD:]], 1))
            return out.double().sort(-1).values * ys.unsqueeze(-1) + ym.unsqueeze(-1)

    def sample_pool_labels(self, state, pool_x, n=1, generator=None):
        """Joint draw of imagined labels for the pool (chunked autoregressive, n_pool_chunks chunks), raw scale / class index."""
        saved = self.n_chunks
        self.n_chunks = self.n_pool_chunks
        try:
            with torch.no_grad():
                d = self._fantasy_draws(state, state["X"], state["y"], pool_x, n, generator)[0]  # [n, N_pool]
                return d.long() if self.task == "classification" else d
        finally:
            self.n_chunks = saved

    def rollout_draws(self, state, eval_x, extra_x, extra_y, weights, n_draws=None, differentiable=False, generator=None):
        """Joint draws over eval_x after the pseudo-update, for the 'risk' objective: [G, S, M] labels (regression) or
        [G, S, M, C] class probabilities given the completed context (classification).  weights: [G, n] binary (hard) or
        fractional (relax='weighted'), or [G, b, n] slots (relax='avg_input').  The pseudo-update uses D's preprocessing
        statistics as in rollout_target_mean; the fantasised evaluation chunks enter as ordinary rows."""
        assert self.relax != "avg_emb", "the risk objective is not implemented for relax='avg_emb'"
        dev, S, G = self.device, int(n_draws or self.n_draws), weights.shape[0]
        cls, C = self.task == "classification", self.n_classes
        per = extra_y.dim() == 2 and extra_y.shape[0] == G
        ey = (extra_y if (per or extra_y.dim() == 1) else extra_y[0]).to(dev).float()
        X, y, keep, stats = state["X"], state["y"], state["keep"], state["stats"]
        ex = eval_x.to(dev)
        M = ex.shape[0]
        W = weights.to(dev, torch.float32)
        cw = None
        if W.dim() == 3:  # slots
            extra_x, ey, W = self._top_slots(extra_x, ey, per, W)
            ctx_x, ctx_y, soft = self._slot_context(state, extra_x, ey, per, W)
        else:
            if not differentiable:  # hard batches: per-rollout contexts of D plus the selected items
                sel = [(W[g] > 0).nonzero().squeeze(1) for g in range(G)]
                assert all(s_.numel() == sel[0].numel() for s_ in sel), "hard rollouts need equal batch sizes"
                ex_x = extra_x.to(dev)
                ctx_x = torch.stack([torch.cat([X, ex_x[s_]], 0) for s_ in sel], 0)
                ctx_y = torch.stack([torch.cat([y, (ey[g] if per else ey)[s_]], 0) for g, s_ in enumerate(sel)], 1)
            else:  # fractional update over the soft context
                if self.soft_topm is not None and self.soft_topm < W.shape[1]:
                    top = torch.topk(W, self.soft_topm, dim=1).indices.unique()
                    extra_x, ey, W = extra_x[top.cpu()] if extra_x.device.type == "cpu" else extra_x[top], (ey[:, top] if per else ey[top]), W[:, top]
                ctx_x = torch.cat([X, extra_x.to(dev)], 0).unsqueeze(0).expand(G, -1, -1)
                ctx_y = torch.cat([y.unsqueeze(1).expand(-1, G), ey.t()], 0) if per else torch.cat([y, ey], 0).unsqueeze(1).expand(-1, G)
                cw = torch.cat([torch.ones(X.shape[0], G, device=dev), W.t()], 0)  # [N, G]
            soft = F.one_hot(ctx_y.t().long(), C).float() if cls else None  # [G, N, C]
        Btot = G * S

        def rep(t, dim):  # repeat each rollout's tensor S times along `dim` (the rollout dimension)
            sh = list(t.shape)
            return t.unsqueeze(dim + 1).expand(*sh[: dim + 1], S, *sh[dim + 1:]).reshape(*sh[:dim], Btot, *sh[dim + 1:])
        cx, cy = rep(ctx_x, 0), rep(ctx_y, 1)
        cwb = rep(cw, 1) if cw is not None else None
        sl = rep(soft, 0) if cls else None
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(_draw_seed(generator)))
        fant = torch.zeros(Btot, M, device=dev, dtype=torch.float64)
        saved, self.checkpoint = self.checkpoint, differentiable or self.checkpoint
        saved_gb, self.grad_batch = self.grad_batch, (8 if differentiable else self.grad_batch)
        ac = torch.autocast(dev.type, dtype=self.sample_dtype or torch.float16, enabled=(not differentiable) and self.sample_dtype is not None)
        try:
            for idx in torch.chunk(perm.to(dev), self.n_chunks):
                qx = ex[idx]
                with ac:
                    q = self.forward_raw(cx, cy, qx, keep, ctx_w=cwb, grad=differentiable, stats=stats, soft_y=sl)
                if cls:
                    g = -torch.log(-torch.log(torch.rand(q.shape, device=dev, generator=generator).clamp_(1e-20, 1)))
                    hard = F.one_hot((torch.log(q.clamp_min(1e-12)) + g).argmax(-1), C).to(q)
                    if differentiable:  # straight-through: one-hot forward, softmax gradient
                        sf = torch.softmax((torch.log(q.clamp_min(1e-12)) + g) / 0.5, -1)
                        hard = hard + sf - sf.detach()
                    sl = torch.cat([sl, hard.float()], 1)
                    cy = torch.cat([cy, hard.argmax(-1).t().float()], 0)
                else:
                    z = self._sample(q, generator)  # [Btot, |idx|], differentiable in the quantiles
                    fant = fant.index_copy(1, idx, z)
                    cy = torch.cat([cy, z.t().float()], 0)
                cx = torch.cat([cx, qx.unsqueeze(0).expand(Btot, -1, -1)], 1)
                if cwb is not None:
                    cwb = torch.cat([cwb, torch.ones(idx.numel(), Btot, device=dev)], 0)
            if cls:
                with ac:
                    pr = self.forward_raw(cx, cy, ex, keep, ctx_w=cwb, grad=differentiable, stats=stats, soft_y=sl)
                return pr.float().view(G, S, M, C)
            return fant.float().view(G, S, M)
        finally:
            self.checkpoint, self.grad_batch = saved, saved_gb
