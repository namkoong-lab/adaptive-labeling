#!/usr/bin/env python
"""Experiment 1: high-dimensional synthetic GP regression (input_dim in {1,5,10,20,50,100}).

Same construction as the paper's synthetic experiment (51 clusters, 100 initial labels from one
cluster, pool 500, eval 285, T=10 batches of b=5, GP UQ module with the true hyperparameters),
generalized to R^d with the geometry-preserving scaling described in
adaptive_sampling_lib.data.make_synthetic_gp_problem.

Usage:
  python run.py --method pathwise --input_dim 10 --seed 0 --n_steps 100 --G 10 --gpu 0
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
from adaptive_sampling_lib.uq_gp import GPModule  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("gp_high_dim")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--input_dim", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="training seed (policy optimisation / rollouts)")
    p.add_argument("--seed_dataset", type=int, default=0)
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_pool", type=int, default=500)
    p.add_argument("--n_anchor", type=int, default=51)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=100, help="policy-gradient steps per epoch")
    p.add_argument("--G", type=int, default=10, help="rollouts per gradient estimate")
    p.add_argument("--n_mc", type=int, default=100, help="posterior draws per rollout")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--outputscale", type=float, default=0.69)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--n_track", type=int, default=1000)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_root", type=str, default=RESULTS_ROOT)
    args = p.parse_args()

    dev = pick_device(args.gpu)
    seed_all(args.seed_dataset)
    prob = make_synthetic_gp_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor,
                                     args.lengthscale, args.outputscale, args.noise_std).to(dev)
    # predictor trained once per (dataset seed, dim): identical across methods and training seeds
    predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(predictor, prob.eval_x, "regression")
    target = MSETarget(psi, noise_var=args.noise_std ** 2)
    truth = oracle_target(prob, psi, args.lengthscale, args.outputscale, args.noise_std)  # denoised target value (generating GP known)
    uq = GPModule(args.lengthscale, args.outputscale, args.noise_std ** 2, 0.0, 1e-4, device=dev)
    seed_all(args.seed)
    out_dir = os.path.join(args.out_root, args.tag, f"dim{args.input_dim}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
