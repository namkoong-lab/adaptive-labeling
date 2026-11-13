#!/usr/bin/env python
"""Experiment 4: classification with the population log loss of a fixed classifier as the target,
TabPFN (v2.5 classifier) as the UQ module.

Target: Q(theta) = E_x sum_c theta_c(x) (-log psi_c(x)), the expected cross-entropy of the fixed
classifier psi (an MLP trained on the biased initial labels) under the latent class probabilities
theta; its posterior mean is available in closed form from TabPFN's predictive, and the planners
use the Proposition-1 gain (squared change of the posterior mean of Q).

Datasets (cluster selection bias: 51 k-means clusters, 100 initial labels from one cluster,
pool 500 and evaluation set 285 from the whole population):
  eicu          : eICU in-hospital mortality (10 most important features, ~10% positives)
  magic         : OpenML 1120 MagicTelescope (10 features, 35% positives)
  credit        : OpenML 42477 default-of-credit-card-clients (23 features, 22% positives)

Usage:
  python run.py --dataset eicu --method reinforce_baseline --seed 0 --n_steps 50 --gpu 0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_sampling_lib.data import make_eicu_classification_problem, make_mimic_problem, make_openml_problem, make_synthetic_logit_problem  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.targets import LogLossTarget  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

RESULTS_ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/tabpfn_classification"
OPENML = {"magic": 1120, "credit": 42477, "phoneme": 1489, "electricity": 151}


def build_problem(args, dev):
    kw = dict(n_train=args.n_train, n_pool=args.n_pool, n_eval=args.n_eval * args.eval_scale, n_clusters=args.n_clusters,
              pool_eval_layout=args.layout, n_eval_clusters=args.n_eval_clusters, eval_scale=args.eval_scale)
    if args.dataset == "eicu":
        prob = make_eicu_classification_problem(args.seed_dataset, n_features=args.n_features, **kw)
    elif args.dataset in OPENML:
        prob = make_openml_problem(OPENML[args.dataset], "classification", args.seed_dataset, **kw)
    elif args.dataset == "mimic":  # MIMIC-IV in-hospital mortality, first-24h ICU vitals (adaptive_sampling_lib.mimic_cohort)
        prob = make_mimic_problem(args.seed_dataset, "classification", n_features=args.n_features, **kw)
    elif args.dataset == "synthetic_logit":  # known ground truth: p(y=1|x) = sigmoid(GP draw) on the clustered GMM layout
        prob = make_synthetic_logit_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor, outputscale=args.outputscale,
                                            layout="gmm", center_scale=args.center_scale, cluster_std=args.cluster_std, lengthscale_per_sqrt_d=args.ls_per_sqrt_d)
    else:
        raise ValueError(args.dataset)
    return prob.to(dev)


def build_uq(args, prob, dev):
    if args.uq == "tabpfn":
        from adaptive_sampling_lib.uq_tabpfn import TabPFNModule

        return TabPFNModule("classification", device=dev, n_draws=args.n_mc, n_chunks=args.n_chunks, n_pool_chunks=args.n_pool_chunks,
                            autocast=not args.fp32, checkpoint=True, draw_type="mean", n_classes=2,
                            soft_topm=(None if args.soft_topm <= 0 else args.soft_topm), input_transform=args.input_transform)
    if args.uq == "ensemble":
        from adaptive_sampling_lib.uq_ensemble import EnsemblePlusModule

        return EnsemblePlusModule(prob.input_dim, (50, 50), 10, alpha="auto", prior_target_std=2.0, n_inner_steps=200, inner_lr=0.1,
                                  weight_decay=0.1, device=dev, n_classes=2, calib_x=torch.cat([prob.eval_x, prob.pool_x], 0))
    raise ValueError(args.uq)


def add_args(p):
    p.add_argument("--dataset", default="eicu", choices=["eicu", "mimic", "synthetic_logit"] + list(OPENML))
    p.add_argument("--input_dim", type=int, default=10, help="synthetic_logit only")
    p.add_argument("--n_anchor", type=int, default=51, help="synthetic_logit only")
    p.add_argument("--outputscale", type=float, default=4.0, help="synthetic_logit: variance of the latent logit GP")
    p.add_argument("--center_scale", type=float, default=0.5)
    p.add_argument("--cluster_std", type=float, default=0.5)
    p.add_argument("--ls_per_sqrt_d", type=float, default=0.5)
    p.add_argument("--uq", default="tabpfn", choices=["tabpfn", "ensemble"])
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_dataset", type=int, default=0)
    p.add_argument("--n_features", type=int, default=10, help="eICU: keep the n most important RF features")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_pool", type=int, default=500)
    p.add_argument("--n_eval", type=int, default=285)
    p.add_argument("--n_clusters", type=int, default=51)
    p.add_argument("--layout", type=str, default="uniform", choices=["uniform", "skewed"],
                   help="skewed: evaluation set from n_eval_clusters clusters, pool from the whole population")
    p.add_argument("--n_eval_clusters", type=int, default=8)
    p.add_argument("--eval_scale", type=int, default=1, help="skewed layout: multiply the evaluation set (285 -> 285*eval_scale) keeping its heavy/light structure")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--n_mc", type=int, default=16)
    p.add_argument("--n_track", type=int, default=128)
    p.add_argument("--n_chunks", type=int, default=4)
    p.add_argument("--n_pool_chunks", type=int, default=5)
    p.add_argument("--soft_topm", type=int, default=50)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--objective", type=str, default="gain", choices=["gain", "gain_paired", "risk"])
    p.add_argument("--grad_chunk", type=int, default=2, help="pathwise rollouts per backward pass (memory)")
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


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    dev = pick_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(args.seed_dataset)
    prob = build_problem(args, dev)
    print(prob.summary(), "train pos rate", float(prob.train_y.float().mean()), "eval pos rate", float(prob.eval_y.float().mean()), flush=True)
    predictor = train_predictor(prob.train_x, prob.train_y, "classification", seed=42, device=dev, n_classes=2)
    log_psi = predictor_outputs(predictor, prob.eval_x, "classification")
    target = LogLossTarget(log_psi)
    uq = build_uq(args, prob, dev)
    truth = None
    if prob.eval_p is not None:  # synthetic: exact population log loss of the fixed classifier
        pe = torch.stack([1 - prob.eval_p, prob.eval_p], -1).to(log_psi)
        truth = {"q_oracle": float(target.mean_from_probs(pe)), "q_oracle_std": 0.0, "q_real": target.true_value(prob.eval_y)}
        print("truth:", truth, flush=True)
    seed_all(args.seed)
    dname = args.dataset + (f"_d{args.input_dim}" if args.dataset.startswith("synthetic") else "")
    out_dir = os.path.join(args.out_root, args.tag, f"{args.uq}_{dname}" + ("_skewed" if args.layout == "skewed" else "") + (f"_x{args.eval_scale}" if args.eval_scale != 1 else "") + (f"_tx{args.input_transform}" if args.input_transform != "none" else "") + ("_fp32" if args.fp32 else ""))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose, objective=args.objective, grad_chunk=args.grad_chunk, imagined_labels=args.imagined_labels)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
