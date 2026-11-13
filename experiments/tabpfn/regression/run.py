#!/usr/bin/env python
"""Experiment 3: regression with TabPFN (v2.5) as the UQ module.

Datasets:
  synthetic_gmm : d-dimensional GP functions on a compact 51-cluster Gaussian-mixture layout
                  (moderate covariate shift; dims 5/10/20/50 give the high-dim experiments)
  synthetic_paper: the paper's 1-D 51-cluster layout (extreme shift for a foundation model)
  eicu          : the paper's eICU regression split (RF mortality probability as the label)
  california    : California housing (log median value) with k-means-cluster selection bias

Usage:
  python run.py --dataset eicu --method reinforce_baseline --seed 0 --n_steps 100 --G 10 --gpu 0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from adaptive_sampling_lib.data import (load_eicu_regression_problem, make_california_problem,  # noqa: E402
                                        make_mimic_problem, make_openml_problem, make_synthetic_gp_problem)

OPENML_REG = {"bike": (42712, True), "wine": (40498, False), "elevators": (216, False), "cpu_act": (573, False)}
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.oracle_truth import oracle_target  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("tabpfn_regression")


def build_problem(args, dev):
    if args.dataset == "synthetic_gmm":
        prob = make_synthetic_gp_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor,
                                         outputscale=args.outputscale, noise_std=args.noise_std, layout="gmm",
                                         center_scale=args.center_scale, cluster_std=args.cluster_std,
                                         lengthscale_per_sqrt_d=args.ls_per_sqrt_d)
    elif args.dataset == "synthetic_paper":
        prob = make_synthetic_gp_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor,
                                         lengthscale=1.0, outputscale=args.outputscale, noise_std=args.noise_std, layout="paper")
    elif args.dataset == "eicu":
        prob = load_eicu_regression_problem()
        if args.n_eval_max and prob.eval_x.shape[0] > args.n_eval_max:  # the csv has 500 eval rows; the paper uses 285
            prob.eval_x, prob.eval_y = prob.eval_x[: args.n_eval_max], prob.eval_y[: args.n_eval_max]
    elif args.dataset == "california":
        prob = make_california_problem(args.seed_dataset, n_train=args.n_train, n_pool=args.n_pool, n_eval=285 * args.eval_scale,
                                       pool_eval_layout=args.layout, n_eval_clusters=args.n_eval_clusters, eval_scale=args.eval_scale)
    elif args.dataset == "eicu_raw":  # eICU regression rebuilt from the raw file (supports the skewed layout)
        from adaptive_sampling_lib.data import make_eicu_regression_clustered_problem
        prob = make_eicu_regression_clustered_problem(args.seed_dataset, n_train=args.n_train, n_pool=args.n_pool, n_eval=285 * args.eval_scale,
                                                      pool_eval_layout=args.layout, n_eval_clusters=args.n_eval_clusters, eval_scale=args.eval_scale)
    elif args.dataset == "mimic_los":  # MIMIC-IV log ICU length of stay from first-24h vitals
        prob = make_mimic_problem(args.seed_dataset, "icu_los", n_features=10, n_train=args.n_train, n_pool=args.n_pool, n_eval=285 * args.eval_scale,
                                  pool_eval_layout=args.layout, n_eval_clusters=args.n_eval_clusters, eval_scale=args.eval_scale)
    elif args.dataset in OPENML_REG:
        did, logt = OPENML_REG[args.dataset]
        prob = make_openml_problem(did, "regression", args.seed_dataset, log_target=logt, n_train=args.n_train, n_pool=args.n_pool, n_eval=285 * args.eval_scale,
                                   pool_eval_layout=args.layout, n_eval_clusters=args.n_eval_clusters, eval_scale=args.eval_scale)
    else:
        raise ValueError(args.dataset)
    return prob.to(dev)


def build_uq(args, prob, dev):
    if args.uq == "tabpfn":
        from adaptive_sampling_lib.uq_tabpfn import TabPFNModule

        y_range = None
        if args.moments == "range":  # plausible label range of the task (exact [0,1] for the eICU probability label)
            ally = torch.cat([prob.train_y, prob.pool_y, prob.eval_y]).float()
            y_range = (0.0, 1.0) if args.dataset in ("eicu", "eicu_raw") else (float(ally.min()), float(ally.max()))
            print("TabPFN moments conditioned on label range", y_range, flush=True)
        return TabPFNModule("regression", device=dev, n_draws=args.n_mc, n_chunks=args.n_chunks, n_pool_chunks=args.n_pool_chunks,
                            autocast=not args.fp32, checkpoint=True, draw_type=args.draw_type,
                            soft_topm=(None if args.soft_topm <= 0 else args.soft_topm), input_transform=args.input_transform, y_range=y_range)
    if args.uq == "gp":
        from adaptive_sampling_lib.uq_gp import GPModule

        if args.dataset == "eicu":  # hyper-parameters fitted in the paper (Appendix F.3)
            return GPModule(0.5, 1.0, 1e-4, mean_constant=0.255, device=dev)
        ls = prob.meta.get("lengthscale", 1.0)
        return GPModule(ls, args.outputscale, args.noise_std ** 2, 0.0, device=dev)
    raise ValueError(args.uq)


def add_args(p):
    p.add_argument("--dataset", default="eicu", choices=["synthetic_gmm", "synthetic_paper", "eicu", "eicu_raw", "california", "mimic_los"] + list(OPENML_REG))
    p.add_argument("--layout", type=str, default="uniform", choices=["uniform", "skewed"],
                   help="skewed: evaluation set from n_eval_clusters clusters, pool from the whole population (not for 'eicu' csvs)")
    p.add_argument("--n_eval_clusters", type=int, default=8)
    p.add_argument("--eval_scale", type=int, default=1, help="skewed layout: multiply the evaluation set (285 -> 285*eval_scale) keeping its heavy/light structure")
    p.add_argument("--uq", default="tabpfn", choices=["tabpfn", "gp"])
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_dataset", type=int, default=0)
    p.add_argument("--input_dim", type=int, default=10)
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_pool", type=int, default=500)
    p.add_argument("--n_anchor", type=int, default=51)
    p.add_argument("--n_eval_max", type=int, default=285)
    p.add_argument("--outputscale", type=float, default=0.69)
    p.add_argument("--noise_std", type=float, default=0.1)
    p.add_argument("--center_scale", type=float, default=0.5)
    p.add_argument("--cluster_std", type=float, default=0.5)
    p.add_argument("--ls_per_sqrt_d", type=float, default=0.5)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--n_mc", type=int, default=16, help="posterior draws per rollout")
    p.add_argument("--n_track", type=int, default=128, help="posterior draws for the tracked metrics")
    p.add_argument("--n_chunks", type=int, default=4)
    p.add_argument("--n_pool_chunks", type=int, default=5)
    p.add_argument("--draw_type", type=str, default=None, choices=[None, "labels", "mean"])
    p.add_argument("--soft_topm", type=int, default=50, help="pool items kept in the soft context (<=0: all)")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--objective", type=str, default="gain", choices=["gain", "gain_paired", "risk"],
                   help="rollout value: 'gain' = squared change of the posterior mean of the target (Prop. 1, closed form for TabPFN); 'risk' = MC posterior variance")
    p.add_argument("--grad_chunk", type=int, default=2, help="pathwise rollouts per backward pass (memory)")
    p.add_argument("--moments", type=str, default="full", choices=["full", "range"],
                   help="range: TabPFN regression moments conditioned on the task's label range (removes heavy bar-distribution tails)")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--input_transform", default="none", choices=["none", "fix", "squash3", "squash10", "asinh"],
                   help="feature transform fitted on the labelled points before TabPFN (result group suffix _tx<name>)")
    p.add_argument("--imagined_labels", default="per_step", choices=["per_step", "per_rollout", "per_step_bank"],
                   help="imagined pool labels per planner step: one vector shared by the G rollouts, or one per rollout")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_root", type=str, default=RESULTS_ROOT)
    return p


def output_path(args):
    """The result JSON of a run (make_jobs.py uses it for the jobs' "## <output>" markers)."""
    dname = args.dataset + (f"_d{args.input_dim}" if args.dataset.startswith("synthetic") else "")
    out_dir = os.path.join(args.out_root, args.tag, f"{args.uq}_{dname}" + ("_skewed" if args.layout == "skewed" else "") + (f"_x{args.eval_scale}" if args.eval_scale != 1 else "") + ("_rng" if args.moments == "range" else "") + (f"_c{args.n_chunks}" if args.n_chunks != 4 else "") + (f"_tx{args.input_transform}" if args.input_transform != "none" else "") + ("_fp32" if args.fp32 else ""))
    return os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    dev = pick_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(args.seed_dataset)
    prob = build_problem(args, dev)
    print(prob.summary(), flush=True)
    predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(predictor, prob.eval_x, "regression")
    noise_var = 0.0 if args.uq == "tabpfn" else (1e-4 if args.dataset.startswith("eicu") else args.noise_std ** 2)
    target = MSETarget(psi, noise_var=noise_var)
    uq = build_uq(args, prob, dev)
    truth = None
    if args.dataset.startswith("synthetic"):  # generating GP known -> denoised target value for the error metric
        truth = oracle_target(prob, psi, prob.meta["lengthscale"], args.outputscale, args.noise_std)
        print("truth:", truth, flush=True)
    seed_all(args.seed)
    out_path = output_path(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose, objective=args.objective, grad_chunk=args.grad_chunk, imagined_labels=args.imagined_labels)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
