"""Compare the REINFORCE and relaxed-top-k gradient estimators at a fixed weight vector.

For a fixed pool-score vector w we estimate, for each number of samples G:

    E[cos(g_hat, g_true)]   -- how well one update points the right way
    E||g_hat - E[g_hat]||^2 -- the estimator's variance (trace of its covariance)

for the estimators

    pg_no_baseline    (1/G) sum_i  L_i          * grad log p(S_i)
    pg_with_baseline  (1/G) sum_i (L_i - Lbar)  * grad log p(S_i)      Lbar = batch mean
    relaxed_tau=<t>   (1/G) sum_i  grad L(soft_k(w; tau))              one series per tau

g_true is grad_w E_{S~p_w}[L(S)], estimated by the score-function identity over a
dedicated reference bank, so it is the gradient of the objective REINFORCE targets.
The relaxed estimator targets a temperature-smoothed objective, so its cosine
plateaus below 1 -- that plateau is the relaxation bias and is reported per tau.

Design notes
------------
* Both estimators use the SAME objective, computed with the custom weighted Cholesky
  GP, so nothing depends on the gpytorch-vs-custom-GP difference between the pipelines.
* Almost all of the per-sample variance in L(S) is Monte-Carlo noise from n_samples
  inside var_l2_loss_custom_gp_estimator, not the subset draw. The bootstrap banks
  therefore keep the pipeline's n_samples (the variance curves must reflect training
  conditions), while the reference banks use n_samples_ref >> n_samples. E_Z[L(S,Z)]
  is the true loss either way, so the reference target is unchanged -- only quieter.
* Reference banks are drawn independently of the bootstrap banks. Sharing them would
  correlate g_hat with g_true and inflate the cosine at large G.
* Banks are cached to --cache_dir keyed by their parameters, so re-running with a
  different --G_list is instant.

Pool labels are the true pool_y (access_to_true_pool_y = True).

Usage:
    python gradient_estimator_diagnostics.py \
        --config_file_path config_sweep_pg_no_baseline.json \
        --predictor_path const:0.0
"""

import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import wandb

import polyadic_sampler_new as polyadic_sampler
import k_subset_sampling
from custom_gp_cholesky import GaussianProcessCholesky, RBFKernel
from variance_l_2_loss import var_l2_loss_custom_gp_estimator
from predictor_network import load_predictor

# categorical slots 1-5 of the validated palette, assigned in fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#191918", "#5c5b56", "#8a8983"


def first(values):
    """wandb sweep configs store every parameter as {'values': [...]}."""
    return values["values"][0] if isinstance(values, dict) and "values" in values else values


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    params = raw.get("parameters", raw)
    return {k: first(v) for k, v in params.items()}


def build_dataset(cfg, device):
    """Regenerate the sweep's dataset (the 'specify_own' GP branch of the runners)."""
    torch.manual_seed(cfg["seed_dataset"])
    np.random.seed(cfg["seed_dataset"])

    mean_module = gpytorch.means.ConstantMean()
    base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_module.constant = torch.tensor([cfg["dataset_mean_constant"]])
    base_kernel.base_kernel.lengthscale = cfg["dataset_length_scale"]
    base_kernel.outputscale = cfg["dataset_output_scale"]
    likelihood.noise_covar.noise = cfg["dataset_noise_std"] ** 2

    dumi_x = torch.randn(10, cfg["input_dim"])
    dumi_y = torch.zeros(10)
    dataset_model = polyadic_sampler.CustomizableGPModel(
        dumi_x, dumi_y, mean_module, base_kernel, likelihood)

    sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(
        no_train_points=cfg["no_train_points"], no_test_points=cfg["no_test_points"],
        no_pool_points=cfg["no_pool_points"], model_name=cfg["dataset_model_name"],
        no_anchor_points=cfg["no_anchor_points"], input_dim=cfg["input_dim"],
        stdev_scale=cfg["stdev_scale"], stdev_pool_scale=cfg["stdev_pool_scale"],
        scaling_factor=cfg["scaling_factor"], scale_by_input_dim=cfg["scale_by_input_dim"],
        model=dataset_model, stdev_blr_w=cfg["stdev_blr_w"], stdev_blr_noise=cfg["stdev_blr_noise"],
        logits=cfg["logits"], if_logits=cfg["if_logits"],
        if_logits_only_pool=cfg["if_logits_only_pool"], plot_folder="none")

    train_x, train_y, test_x, test_y, pool_x, pool_y, _, _ = \
        polyadic_sampler.set_data_parameters_and_generate(sampler_cfg)
    return [t.to(device) for t in (train_x, train_y, pool_x, pool_y, test_x, test_y)]


class Objective:
    """L(S) and its relaxed counterpart on the custom weighted-Cholesky GP."""

    def __init__(self, cfg, train_x, train_y, pool_x, pool_y, test_x, predictor, device):
        self.device = device
        self.train_x, self.train_y = train_x, train_y
        self.pool_x, self.pool_y = pool_x, pool_y
        self.test_x = test_x
        self.predictor = predictor
        self.noise_var = cfg["noise_var"]
        self.stab = cfg.get("stabilizing_constant", 1e-5)
        self.n_samples = cfg["n_samples"]
        self.k = cfg["batch_size_query"]

        kernel = RBFKernel(length_scale=cfg["length_scale"],
                           output_scale=cfg["output_scale"]).to(device)
        self.gp = GaussianProcessCholesky(kernel=kernel).to(device)

        self.x_full = torch.cat([train_x, pool_x], dim=0)
        self.y_full = torch.cat([train_y, pool_y], dim=0)
        self.n_train = train_x.size(0)

    def _loss_from(self, x_gp, y_gp, w_gp, n_samples):
        mu, cov = self.gp(x_gp, y_gp, w_gp, self.test_x, self.stab, self.noise_var)
        _, var_loss = var_l2_loss_custom_gp_estimator(
            mu, cov, self.noise_var, self.test_x, self.predictor, self.device, n_samples)
        return var_loss

    def hard_loss(self, indices, n_samples=None):
        """L(S) for a discrete subset. Reduced GP -- see check_hard_weight_equivalence."""
        x_gp = torch.cat([self.train_x, self.pool_x[indices]], dim=0)
        y_gp = torch.cat([self.train_y, self.pool_y[indices]], dim=0)
        w_gp = torch.ones(x_gp.size(0), device=self.device)
        with torch.no_grad():
            return self._loss_from(x_gp, y_gp, w_gp, n_samples or self.n_samples).item()

    def hard_loss_full_pool(self, indices, n_samples=None):
        """Same quantity via 0/1 weights over the whole pool, as test_smaller_dataset does."""
        hard = torch.zeros(self.pool_x.size(0), device=self.device)
        hard[indices] = 1.0
        w_gp = torch.cat([torch.ones(self.n_train, device=self.device), hard])
        with torch.no_grad():
            return self._loss_from(self.x_full, self.y_full, w_gp, n_samples or self.n_samples).item()

    def relaxed_grad(self, w, subset_op, n_samples=None):
        """grad_w L(soft_k(w)) -- one draw, exactly as train_smaller_dataset computes it."""
        w_param = w.clone().detach().requires_grad_(True)
        soft = subset_op(w_param.unsqueeze(0)).squeeze(0)
        soft = torch.clamp(soft, min=-float("inf"), max=1.0)
        w_gp = torch.cat([torch.ones(self.n_train, device=self.device), soft])
        loss = self._loss_from(self.x_full, self.y_full, w_gp, n_samples or self.n_samples)
        grad, = torch.autograd.grad(loss, w_param)
        return grad.detach(), loss.detach().item()


def check_hard_weight_equivalence(obj, seed=0):
    """Zero-weighting a pool point must equal dropping it. Guards the reduced-GP shortcut."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(obj.pool_x.size(0), generator=g)[:obj.k].to(obj.device)
    torch.manual_seed(seed)
    a = obj.hard_loss(idx)
    torch.manual_seed(seed)
    b = obj.hard_loss_full_pool(idx)
    rel = abs(a - b) / max(abs(b), 1e-12)
    print(f"[check] hard-weight equivalence: reduced={a:.6e} full-pool={b:.6e} rel_diff={rel:.2e}")
    return rel


def log_prob_and_grad(w, indices):
    """The PG log-prob of gp_pipeline_regression_pg.py:255-257, and its gradient wrt w."""
    w_param = w.clone().detach().requires_grad_(True)
    prob = F.softmax(w_param, dim=0)
    log_pr = torch.log(prob[indices]).sum()
    for i in range(indices.size(0)):
        log_pr = log_pr - torch.log(1 - prob[indices[:i]].sum())
    grad, = torch.autograd.grad(log_pr, w_param)
    return grad.detach()


# ---------------------------------------------------------------- bank caching

def cached(cache_dir, kind, meta, builder, refresh=False):
    """Build a sample bank once and reuse it across runs. Key is a hash of its parameters."""
    key = hashlib.md5(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{kind}_{key}.pt")
    if os.path.exists(path) and not refresh:
        blob = torch.load(path, weights_only=False)
        print(f"[cache] hit  {kind:18s} {os.path.basename(path)}  {meta}")
        return blob["data"]
    print(f"[cache] miss {kind:18s} building {meta}")
    data = builder()
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"meta": meta, "data": data}, path)
    print(f"[cache] wrote {path}")
    return data


def build_pg_bank(obj, w, M, n_samples, log_every=5000):
    """M i.i.d. rollouts: the loss and the score-function gradient of each."""
    prob = F.softmax(w, dim=0)
    losses = torch.empty(M, device=obj.device)
    glogp = torch.empty(M, w.numel(), device=obj.device)
    for m in range(M):
        indices = torch.multinomial(prob, obj.k, replacement=False)
        losses[m] = obj.hard_loss(indices, n_samples)
        glogp[m] = log_prob_and_grad(w, indices)
        if log_every and (m + 1) % log_every == 0:
            print(f"  pg bank {m + 1}/{M}", flush=True)
    return losses, glogp


def build_relaxed_bank(obj, w, M, tau, n_samples, device, k, log_every=2000):
    op = k_subset_sampling.SubsetOperator(k, device, tau, False).to(device)
    grads = torch.empty(M, w.numel(), device=device)
    losses = torch.empty(M, device=device)
    for m in range(M):
        grads[m], losses[m] = obj.relaxed_grad(w, op, n_samples)
        if log_every and (m + 1) % log_every == 0:
            print(f"  relaxed(tau={tau}) bank {m + 1}/{M}", flush=True)
    return losses, grads


# ---------------------------------------------------------------- estimators

def reference_gradient(losses, glogp):
    """Centering removes the E[L]*grad-log-p term, which carries nearly all the noise.

    E[(L_i - Lbar) grad log p_i] = (1 - 1/M) * grad_w E[L] for independent draws, so
    dividing by (1 - 1/M) leaves an unbiased estimate at a fraction of the variance.
    """
    M = losses.numel()
    per_sample = (losses - losses.mean()).unsqueeze(-1) * glogp
    g = per_sample.mean(dim=0) / (1.0 - 1.0 / M)
    se = math.sqrt((per_sample - per_sample.mean(dim=0)).pow(2).sum(dim=1).mean().item() / M)
    return g, se


def bootstrap(losses, glogp, relaxed_banks, G, R, chunk, generator):
    """R independent G-sample estimators of each kind, drawn from the bootstrap banks."""
    M, P = glogp.shape
    names = ["pg_no_baseline", "pg_with_baseline"] + [f"relaxed_tau={t}" for t in relaxed_banks]
    out = {n: torch.empty(R, P, device=glogp.device) for n in names}
    for start in range(0, R, chunk):
        r = min(chunk, R - start)
        idx = torch.randint(0, M, (r, G), device=glogp.device, generator=generator)
        L = losses[idx]
        g = glogp[idx]
        out["pg_no_baseline"][start:start + r] = (L.unsqueeze(-1) * g).mean(dim=1)
        adv = L - L.mean(dim=1, keepdim=True)
        out["pg_with_baseline"][start:start + r] = (adv.unsqueeze(-1) * g).mean(dim=1)
        for tau, grads in relaxed_banks.items():
            j = torch.randint(0, grads.shape[0], (r, G), device=grads.device, generator=generator)
            out[f"relaxed_tau={tau}"][start:start + r] = grads[j].mean(dim=1)
    return out


def summarize(est, g_true):
    mean_est = est.mean(dim=0)
    cos_true = F.cosine_similarity(est, g_true.unsqueeze(0).expand_as(est), dim=1)
    cos_self = F.cosine_similarity(est, mean_est.unsqueeze(0).expand_as(est), dim=1)
    variance = (est - mean_est.unsqueeze(0)).pow(2).sum(dim=1).mean()
    denom = mean_est.pow(2).sum()
    return {
        "cos_true_mean": cos_true.mean().item(),
        "cos_true_se": (cos_true.std() / math.sqrt(est.shape[0])).item(),
        "cos_self_mean": cos_self.mean().item(),
        "variance": variance.item(),
        "mean_norm": mean_est.norm().item(),
        "rel_variance": (variance / denom).item() if denom > 0 else float("nan"),
    }


# ---------------------------------------------------------------- output

def plot(results, G_list, series, colors, bias, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax in axes:
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d5d4cf")
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.set_xscale("log")
        ax.set_xticks(G_list)
        ax.set_xticklabels([str(g) for g in G_list])
        ax.minorticks_off()
        ax.set_xlabel("G (samples per update)", color=INK_SECONDARY, fontsize=10)

    for name in series:
        color = colors[name]
        cos = [results[g][name]["cos_true_mean"] for g in G_list]
        se = [results[g][name]["cos_true_se"] for g in G_list]
        var = [results[g][name]["variance"] for g in G_list]
        axes[0].errorbar(G_list, cos, yerr=se, color=color, linewidth=2,
                         marker="o", markersize=5, capsize=3, label=name)
        axes[1].plot(G_list, var, color=color, linewidth=2, marker="o",
                     markersize=5, label=name)
        if name in bias:   # asymptote the relaxed curves converge to
            axes[0].axhline(bias[name], color=color, linewidth=1, linestyle="--", alpha=0.55)

    axes[0].axhline(0.0, color=INK_MUTED, linewidth=1, linestyle=":")
    if 1 in G_list:
        axes[0].annotate("G=1: baseline\nestimator " + r"$\equiv 0$", xy=(1, 0.0),
                         xytext=(1.06, 0.05), fontsize=8, color=colors["pg_with_baseline"])
    axes[0].set_ylabel("E[cos(estimate, true gradient)]", color=INK_SECONDARY, fontsize=10)
    axes[0].set_title("Alignment with the true gradient  (dashed = relaxation bias ceiling)",
                      color=INK_PRIMARY, fontsize=11, loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"E$\|\hat{g} - E[\hat{g}]\|^2$", color=INK_SECONDARY, fontsize=10)
    axes[1].set_title("Estimator variance", color=INK_PRIMARY, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file_path", required=True)
    p.add_argument("--predictor_path", default="const:0.0")
    p.add_argument("--G_list", default="1,2,4,8,10,100,1000")
    p.add_argument("--taus", default="0.1,0.5,10", help="relaxation temperatures to compare")
    p.add_argument("--M", type=int, default=20000, help="PG bootstrap bank")
    p.add_argument("--M_ref", type=int, default=10000, help="PG reference bank (low noise)")
    p.add_argument("--M_relaxed", type=int, default=8000, help="relaxed bootstrap bank, per tau")
    p.add_argument("--M_relaxed_ref", type=int, default=2000, help="relaxed reference bank, per tau")
    p.add_argument("--n_samples_ref", type=int, default=4000,
                   help="posterior samples in the loss estimator for the reference banks")
    p.add_argument("--R", type=int, default=4000, help="bootstrap draws per G")
    p.add_argument("--chunk", type=int, default=16, help="bootstrap draws held in memory at once")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dir", default="grad_diag_cache")
    p.add_argument("--refresh", action="store_true", help="ignore cached banks and rebuild")
    p.add_argument("--out_prefix", default="grad_diag")
    args = p.parse_args()

    G_list = [int(g) for g in args.G_list.split(",")]
    taus = [float(t) for t in args.taus.split(",")]
    device = args.device
    wandb.init(mode="disabled")   # polyadic_sampler_new logs its dataset plots to wandb

    # float64 for the GP math: with only the init train set conditioned on, the 285-point
    # test covariance is too ill-conditioned for a float32 Cholesky at the pipeline's 1e-5
    # jitter. The predictor stays float32, as in the pipelines.
    torch.set_default_dtype(torch.float64)

    cfg = load_config(args.config_file_path)
    tag = os.path.splitext(os.path.basename(args.config_file_path))[0]
    print(f"[config] pool={cfg['no_pool_points']} k={cfg['batch_size_query']} "
          f"dim={cfg['input_dim']} n_samples={cfg['n_samples']} taus={taus}")

    train_x, train_y, pool_x, pool_y, test_x, test_y = build_dataset(cfg, device)
    predictor = load_predictor(args.predictor_path, device=device)
    predictor.float().eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obj = Objective(cfg, train_x, train_y, pool_x, pool_y, test_x, predictor, device)
    check_hard_weight_equivalence(obj)

    pool_size = pool_x.size(0)
    w = torch.full([pool_size], math.log(1.0 / pool_size), device=device)  # the shared init
    base = {"cfg": tag, "seed": args.seed, "k": cfg["batch_size_query"], "pool": pool_size}

    losses, glogp = cached(
        args.cache_dir, "pg_bootstrap",
        {**base, "M": args.M, "n_samples": cfg["n_samples"]},
        lambda: build_pg_bank(obj, w, args.M, cfg["n_samples"]), args.refresh)

    ref_losses, ref_glogp = cached(
        args.cache_dir, "pg_reference",
        {**base, "M": args.M_ref, "n_samples": args.n_samples_ref, "role": "ref"},
        lambda: build_pg_bank(obj, w, args.M_ref, args.n_samples_ref), args.refresh)

    relaxed_banks, bias = {}, {}
    for tau in taus:
        _, grads = cached(
            args.cache_dir, "relaxed_bootstrap",
            {**base, "M": args.M_relaxed, "tau": tau, "n_samples": cfg["n_samples"]},
            lambda tau=tau: build_relaxed_bank(obj, w, args.M_relaxed, tau, cfg["n_samples"],
                                               device, cfg["batch_size_query"]), args.refresh)
        relaxed_banks[tau] = grads
        _, ref_grads = cached(
            args.cache_dir, "relaxed_reference",
            {**base, "M": args.M_relaxed_ref, "tau": tau,
             "n_samples": args.n_samples_ref, "role": "ref"},
            lambda tau=tau: build_relaxed_bank(obj, w, args.M_relaxed_ref, tau,
                                               args.n_samples_ref, device,
                                               cfg["batch_size_query"]), args.refresh)
        bias[tau] = ref_grads

    g_true, g_true_se = reference_gradient(ref_losses, ref_glogp)
    ratio = g_true_se / g_true.norm().item()
    print(f"\n[ref] ||g_true|| = {g_true.norm().item():.6e}  (se {g_true_se:.2e}, "
          f"noise/signal {ratio:.3f})")
    if ratio > 0.2:
        print("[warn] reference gradient is not resolved -- raise --M_ref or --n_samples_ref")

    bias_cos = {}
    for tau, ref_grads in bias.items():
        mean = ref_grads.mean(dim=0)
        se = math.sqrt((ref_grads - mean).pow(2).sum(dim=1).mean().item() / ref_grads.shape[0])
        c = F.cosine_similarity(mean, g_true, dim=0).item()
        bias_cos[f"relaxed_tau={tau}"] = c
        print(f"[bias] tau={tau:<5} cos(E[g_relaxed], g_true) = {c:+.4f}   "
              f"||E[g]|| = {mean.norm().item():.3e} (noise/signal {se / mean.norm().item():.3f})")

    series = ["pg_no_baseline", "pg_with_baseline"] + [f"relaxed_tau={t}" for t in taus]
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(series)}

    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    results, rows = {}, []
    for G in G_list:
        est = bootstrap(losses, glogp, relaxed_banks, G, args.R, args.chunk, gen)
        results[G] = {name: summarize(e, g_true) for name, e in est.items()}
        for name in series:
            s = results[G][name]
            rows.append([G, name, s["cos_true_mean"], s["cos_true_se"], s["cos_self_mean"],
                         s["variance"], s["rel_variance"], s["mean_norm"]])

    header = (f"{'G':>5}  {'estimator':<20} {'cos(true)':>11} {'+-se':>8} {'cos(own)':>10} "
              f"{'variance':>12} {'rel var':>10} {'||E[g]||':>11}")
    print("\n" + header)
    print("-" * len(header))
    for G, name, cos, se, cos_self, var, relvar, norm in rows:
        print(f"{G:>5}  {name:<20} {cos:>11.4f} {se:>8.4f} {cos_self:>10.4f} "
              f"{var:>12.4e} {relvar:>10.2f} {norm:>11.4e}")

    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w") as f:
        f.write("G,estimator,cos_true,cos_true_se,cos_own_mean,variance,rel_variance,mean_norm\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"[saved] {csv_path}")

    plot(results, G_list, series, colors, bias_cos, f"{args.out_prefix}.png")


if __name__ == "__main__":
    main()
