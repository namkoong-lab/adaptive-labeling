#!/usr/bin/env python
"""GP regression: one run of a planner or baseline (10 batches of 5 labels).

--dataset synthetic (default): the paper's synthetic construction in R^d (input_dim in {1, 5, 10, 20, 50, 100}): 51
clusters, 100 initial labels from one cluster, a pool of 500 and an evaluation set of 285, labels drawn from a GP; the
belief is a GP with the generating hyperparameters (lengthscale 1, outputscale 0.69, noise sd 0.1), and the error metric
uses the denoised target value of the generating GP.  The geometry-preserving scaling to R^d is described in
adaptive_sampling_lib.data.make_synthetic_gp_problem.
--dataset eicu: the paper's eICU split (100 initial labels from one k-means cluster, a pool of 388 and the first 285 of
the 500 evaluation rows, 10 standardised features; label = random-forest predicted in-hospital mortality probability) with the GP hyperparameters
fitted in the paper (lengthscale 0.5, outputscale 1, noise variance 1e-4, prior mean 0.255); the error metric uses the
evaluation labels.

  python run.py --method pathwise --input_dim 10 --seed 1 --seed_dataset 1 --n_steps 100 --tau 0.5 --gpu 0
  python run.py --dataset eicu --method reinforce_baseline --seed 3 --n_steps 1000 --gpu 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from adaptive_sampling_lib.data import make_synthetic_gp_problem, load_eicu_regression_problem  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.oracle_truth import oracle_target  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.uq_gp import GPModule  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("gp_high_dim")
# GP hyperparameters per dataset: (lengthscale, outputscale, noise_std, prior mean)
HYPER = {"synthetic": (1.0, 0.69, 0.1, 0.0),  # the generating GP
         "eicu": (0.5, 1.0, 0.01, 0.255)}  # fitted in the paper (Appendix F.3)


def add_args(p):
    p.add_argument("--method", required=True)
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "eicu"])
    p.add_argument("--input_dim", type=int, default=1, help="synthetic only")
    p.add_argument("--seed", type=int, default=0, help="training seed (policy optimisation / rollouts)")
    p.add_argument("--seed_dataset", type=int, default=0, help="synthetic only: the GP function draw and the cluster layout")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_pool", type=int, default=500)
    p.add_argument("--n_anchor", type=int, default=51)
    p.add_argument("--n_eval_max", type=int, default=285, help="eicu: the evaluation csv has 500 rows; the paper uses the first 285")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--b", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=100, help="policy-gradient steps per epoch")
    p.add_argument("--G", type=int, default=10, help="rollouts per gradient estimate")
    p.add_argument("--n_mc", type=int, default=100, help="posterior draws per rollout")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lengthscale", type=float, default=None, help="default: HYPER[dataset]")
    p.add_argument("--outputscale", type=float, default=None, help="default: HYPER[dataset]")
    p.add_argument("--noise_std", type=float, default=None, help="default: HYPER[dataset]")
    p.add_argument("--mean_constant", type=float, default=None, help="GP prior mean; default: HYPER[dataset]")
    p.add_argument("--n_track", type=int, default=1000)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out_root", type=str, default=RESULTS_ROOT)
    return p


def resolve_hyper(args):
    """Fill the GP hyperparameters left unset with the dataset's values (in place)."""
    for name, v in zip(("lengthscale", "outputscale", "noise_std", "mean_constant"), HYPER[args.dataset]):
        if getattr(args, name) is None:
            setattr(args, name, v)
    return args


def output_path(args):
    """The result JSON of a run (make_jobs.py uses it for the jobs' "## <output>" markers)."""
    sub = f"dim{args.input_dim}" if args.dataset == "synthetic" else args.dataset
    return os.path.join(args.out_root, args.tag, sub, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")


def build(args, dev):
    """-> (problem, target, GP belief, truth) of a run; truth is the denoised target value (synthetic) or None."""
    resolve_hyper(args)
    seed_all(args.seed_dataset)
    if args.dataset == "synthetic":
        prob = make_synthetic_gp_problem(args.seed_dataset, args.input_dim, args.n_train, args.n_pool, args.n_anchor,
                                         args.lengthscale, args.outputscale, args.noise_std).to(dev)
    else:
        prob = load_eicu_regression_problem()
        if args.n_eval_max and prob.eval_x.shape[0] > args.n_eval_max:
            prob.eval_x, prob.eval_y = prob.eval_x[: args.n_eval_max], prob.eval_y[: args.n_eval_max]
        prob = prob.to(dev)
    # predictor trained once per problem: identical across methods and training seeds
    predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(predictor, prob.eval_x, "regression")
    target = MSETarget(psi, noise_var=args.noise_std ** 2)
    truth = None
    if args.dataset == "synthetic":  # generating GP known -> denoised target value for the error metric
        truth = oracle_target(prob, psi, args.lengthscale, args.outputscale, args.noise_std)
    uq = GPModule(args.lengthscale, args.outputscale, args.noise_std ** 2, args.mean_constant, 1e-4, device=dev)
    return prob, target, uq, truth


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    dev = pick_device(args.gpu)
    prob, target, uq, truth = build(args, dev)
    seed_all(args.seed)
    out_path = output_path(args)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
