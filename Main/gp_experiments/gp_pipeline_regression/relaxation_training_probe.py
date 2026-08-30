"""Why can more meta-iterations select a worse batch?

Runs the actual relaxed-top-k optimisation from the uniform init (Adam, the pipeline's
lr and temperature) and, at a set of checkpoints, records what the hard top-k readout
would pick: the resulting L(S) and how many distinct anchor clusters the batch covers.

The dataset places 5 near-identical pool points on each of anchors 1..50 (plus 250
redundant points on anchor 0, where the training set lives), so a batch drawn from
5 different anchors is far more informative than 5 points of one anchor. torch.topk
scores items independently, which is where diversity can be lost.

    python relaxation_training_probe.py --config_file_path config_sweep_pg_no_baseline.json \
        --tau 0.5 --G 2 --seeds 10
"""

import argparse
import math
import warnings

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import wandb

import gpytorch
import polyadic_sampler_new as polyadic_sampler
import k_subset_sampling
from predictor_network import load_predictor
from gradient_estimator_diagnostics import load_config, Objective


def build_dataset_with_clusters(cfg, device):
    """Same dataset as the runners, but keeping pool_sample_idx (the anchor of each point)."""
    torch.manual_seed(cfg["seed_dataset"])
    np.random.seed(cfg["seed_dataset"])

    mean_module = gpytorch.means.ConstantMean()
    base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_module.constant = torch.tensor([cfg["dataset_mean_constant"]])
    base_kernel.base_kernel.lengthscale = cfg["dataset_length_scale"]
    base_kernel.outputscale = cfg["dataset_output_scale"]
    likelihood.noise_covar.noise = cfg["dataset_noise_std"] ** 2

    dataset_model = polyadic_sampler.CustomizableGPModel(
        torch.randn(10, cfg["input_dim"]), torch.zeros(10),
        mean_module, base_kernel, likelihood)

    sampler_cfg = polyadic_sampler.PolyadicSamplerConfig(
        no_train_points=cfg["no_train_points"], no_test_points=cfg["no_test_points"],
        no_pool_points=cfg["no_pool_points"], model_name=cfg["dataset_model_name"],
        no_anchor_points=cfg["no_anchor_points"], input_dim=cfg["input_dim"],
        stdev_scale=cfg["stdev_scale"], stdev_pool_scale=cfg["stdev_pool_scale"],
        scaling_factor=cfg["scaling_factor"], scale_by_input_dim=cfg["scale_by_input_dim"],
        model=dataset_model, stdev_blr_w=cfg["stdev_blr_w"],
        stdev_blr_noise=cfg["stdev_blr_noise"], logits=cfg["logits"],
        if_logits=cfg["if_logits"], if_logits_only_pool=cfg["if_logits_only_pool"],
        plot_folder="none")

    train_x, train_y, test_x, test_y, pool_x, pool_y, _, pool_sample_idx = \
        polyadic_sampler.set_data_parameters_and_generate(sampler_cfg)
    out = [t.to(device) for t in (train_x, train_y, pool_x, pool_y, test_x, test_y)]
    return out + [pool_sample_idx.to(device)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file_path", required=True)
    p.add_argument("--predictor_path", default="const:0.0")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--G", type=int, default=2)
    p.add_argument("--lr", type=float, default=None, help="defaults to config meta_opt_lr")
    p.add_argument("--checkpoints", default="0,1,2,5,10,20,50")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n_samples_eval", type=int, default=4000,
                   help="posterior samples when scoring the selected batch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    wandb.init(mode="disabled")
    torch.set_default_dtype(torch.float64)

    cfg = load_config(args.config_file_path)
    lr = args.lr if args.lr is not None else cfg["meta_opt_lr"]
    checkpoints = sorted({int(c) for c in args.checkpoints.split(",")})
    n_iter = max(checkpoints)

    train_x, train_y, pool_x, pool_y, test_x, test_y, pool_cluster = \
        build_dataset_with_clusters(cfg, args.device)
    predictor = load_predictor(args.predictor_path, device=args.device)
    predictor.float().eval()

    obj = Objective(cfg, train_x, train_y, pool_x, pool_y, test_x, predictor, args.device)
    k, P = cfg["batch_size_query"], pool_x.size(0)
    op = k_subset_sampling.SubsetOperator(k, args.device, args.tau, False).to(args.device)

    print(f"[setup] tau={args.tau} G={args.G} lr={lr} k={k} pool={P} "
          f"n_iter={n_iter} seeds={args.seeds}")

    # a random-batch reference, and the best single-anchor (fully redundant) batch,
    # to place the numbers on a scale
    torch.manual_seed(12345)
    rand_losses = [obj.hard_loss(torch.randperm(P, device=args.device)[:k], args.n_samples_eval)
                   for _ in range(20)]
    print(f"[scale] random batch L = {np.mean(rand_losses):.6e} +- "
          f"{np.std(rand_losses) / math.sqrt(len(rand_losses)):.1e}")

    record = {c: {"loss": [], "clusters": []} for c in checkpoints}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        w = torch.full([P], math.log(1.0 / P), requires_grad=True, device=args.device)
        opt = torch.optim.Adam([w], lr=lr, weight_decay=cfg["meta_opt_weight_decay"])

        for it in range(n_iter + 1):
            if it in checkpoints:
                idx = torch.topk(w.detach(), k).indices
                record[it]["loss"].append(obj.hard_loss(idx, args.n_samples_eval))
                record[it]["clusters"].append(len(set(pool_cluster[idx].tolist())))
            if it == n_iter:
                break
            opt.zero_grad()
            for _ in range(args.G):                     # matches train_smaller_dataset
                grad, _l = obj.relaxed_grad(w, op)
                w.grad = grad / args.G if w.grad is None else w.grad + grad / args.G
            opt.step()
        print(f"  seed {seed} done", flush=True)

    print(f"\n{'iters':>6} {'L(top-k)':>14} {'+-se':>10} {'distinct anchors':>18} "
          f"{'all-in-one-anchor':>18}")
    print("-" * 70)
    for c in checkpoints:
        L = np.array(record[c]["loss"])
        cl = np.array(record[c]["clusters"])
        print(f"{c:>6} {L.mean():>14.6e} {L.std() / math.sqrt(len(L)):>10.1e} "
              f"{cl.mean():>18.2f} {(cl == 1).mean():>18.0%}")


if __name__ == "__main__":
    main()
