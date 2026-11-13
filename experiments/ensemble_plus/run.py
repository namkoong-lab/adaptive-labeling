#!/usr/bin/env python
"""Experiment 2: Ensemble+ UQ module in a multi-step setting (T decision epochs, batch b), with
REINFORCE (with/without baseline), the smoothed pathwise estimator (differentiable unrolled Adam
inner loop, fully vectorized over ensemble members and rollouts) and the active-learning baselines.

Default setting ("complex"): synthetic GP data in R^d with 11 clusters, 20 initial labels from one
cluster, pool 100, eval 285-ish, T=5 epochs of b=5 (vs. the paper's proof of concept: 4 clusters,
pool 10, one batch of 2).

Usage:
  python run.py --method pathwise --seed 0 --n_steps 50 --G 10 --gpu 0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from adaptive_sampling_lib.data import make_synthetic_gp_problem  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.oracle_truth import oracle_target  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.uq_ensemble import EnsemblePlusModule  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("ensemble_multistep")


def build_problem(args, dev):
    prob = make_synthetic_gp_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor,
                                     args.lengthscale, args.outputscale, args.noise_std,
                                     eval_per_cluster=args.eval_per_cluster, layout=args.layout,
                                     center_scale=args.center_scale, cluster_std=args.cluster_std).to(dev)
    return prob


def build_uq(args, prob, dev):
    prior_std = args.prior_std if args.prior_std_mult <= 0 else args.prior_std_mult * prob.train_y.std().item()
    return EnsemblePlusModule(prob.input_dim, tuple(args.hidden), args.n_members, alpha=args.alpha,
                              prior_target_std=prior_std, n_inner_steps=args.n_inner, inner_lr=args.inner_lr,
                              weight_decay=args.weight_decay, noise_std=args.noise_std, seed_base=args.seed_base,
                              seed_prior=args.seed_prior, device=dev, anchor_to_init=not args.no_anchor,
                              calib_x=torch.cat([prob.eval_x, prob.pool_x], 0), activation=args.activation,
                              inner_opt=args.inner_opt, momentum=args.momentum, ridge=args.ridge)


def add_args(p):
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_dataset", type=int, default=0)
    p.add_argument("--input_dim", type=int, default=5)
    p.add_argument("--layout", type=str, default="gmm", choices=["gmm", "paper"])
    p.add_argument("--center_scale", type=float, default=0.5)
    p.add_argument("--cluster_std", type=float, default=0.5)
    p.add_argument("--n_train", type=int, default=20)
    p.add_argument("--n_pool", type=int, default=100)
    p.add_argument("--n_anchor", type=int, default=11)
    p.add_argument("--eval_per_cluster", type=int, default=20, help="eval points per heavy cluster (paper layout uses n_train/10)")
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--objective", type=str, default="risk", choices=["risk", "gain"])
    p.add_argument("--grad_chunk", type=int, default=2, help="pathwise rollouts per backward pass (memory: ~G*M*steps otherwise)")
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--outputscale", type=float, default=0.69)
    p.add_argument("--noise_std", type=float, default=0.1)
    # ensemble+
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 50])
    p.add_argument("--n_members", type=int, default=10)
    p.add_argument("--alpha", type=str, default="auto", help="'auto' (calibrate prior std) or a float")
    p.add_argument("--prior_std", type=float, default=0.83, help="target prior std when alpha='auto' and prior_std_mult<=0")
    p.add_argument("--prior_std_mult", type=float, default=3.0, help="target prior std = mult * std(train_y) (<=0: use --prior_std)")
    p.add_argument("--activation", type=str, default="silu", choices=["relu", "silu", "tanh", "softplus", "gelu"],
                   help="smooth activations make the trained ensemble a smooth function of the selection weights")
    p.add_argument("--inner_opt", type=str, default="momentum", choices=["adam", "momentum", "sgd"],
                   help="inner optimiser; heavy-ball GD has a well-conditioned unrolled derivative, Adam does not")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--no_anchor", action="store_true", help="regularise towards zero (original) instead of towards the init")
    p.add_argument("--n_inner", type=int, default=800)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--seed_prior", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_root", type=str, default=RESULTS_ROOT)
    return p


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    if args.alpha != "auto":
        args.alpha = float(args.alpha)
    dev = pick_device(args.gpu)
    seed_all(args.seed_dataset)
    prob = build_problem(args, dev)
    predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(predictor, prob.eval_x, "regression")
    target = MSETarget(psi, noise_var=args.noise_std ** 2)
    truth = oracle_target(prob, psi, prob.meta["lengthscale"], args.outputscale, args.noise_std)  # denoised target value (generating GP known)
    uq = build_uq(args, prob, dev)
    print(f"Ensemble+ alpha={uq.alpha:.3f} members={uq.M} inner_steps={uq.n_steps}", flush=True)
    seed_all(args.seed)
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_members, lr=args.lr, tau=args.tau, verbose=args.verbose, objective=args.objective, grad_chunk=args.grad_chunk)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_members, verbose=True, extra_record={"args": vars(args), "device": str(dev), "alpha_used": uq.alpha}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
