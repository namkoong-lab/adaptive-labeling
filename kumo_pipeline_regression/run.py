#!/usr/bin/env python
"""Regression with Kumo-Tabular (NVIDIA; medium by default) as the UQ module (adaptive_sampling_lib.uq_kumo).

Same problems, predictor, target, planners and arguments as tabpfn_pipeline_regression/run.py (whose build_problem and
add_args are reused); the belief is Kumo-Tabular with the weighted pseudo-update.  Objectives gain, gain_paired and risk; relaxations --kumo_relax (adaptive_sampling_lib.uq_kumo).
--moments range conditions the predictive on the task's label range as for TabPFN; --input_transform and --fp32 do not
apply (Kumo's own preprocessing, float32).  Results: results/kumo_regression/<tag>/kumo-<size>_<dataset>[_skewed][_rng].

Usage:
  python run.py --dataset eicu_raw --layout skewed --moments range --method pathwise --tau 0.5 --seed 1 --seed_dataset 1 --n_steps 50 --gpu 0
"""
import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from adaptive_sampling_lib.oracle_truth import oracle_target  # noqa: E402
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.targets import MSETarget  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

_spec = importlib.util.spec_from_file_location("tabpfn_reg_run", os.path.join(HERE, "..", "tabpfn_pipeline_regression", "run.py"))
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
RESULTS_ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/kumo_regression"


def label_range(args, prob):
    """The planner's label range for --moments range (exact [0,1] for the eICU probability label), as for TabPFN."""
    if args.moments != "range":
        return None
    ally = torch.cat([prob.train_y, prob.pool_y, prob.eval_y]).float()
    return (0.0, 1.0) if args.dataset in ("eicu", "eicu_raw") else (float(ally.min()), float(ally.max()))


def build_uq(args, prob, dev):
    from adaptive_sampling_lib.uq_kumo import KumoModule

    y_range = label_range(args, prob)
    if y_range is not None:
        print("Kumo moments conditioned on label range", y_range, flush=True)
    return KumoModule(args.kumo_size, device=dev, n_draws=args.n_mc, n_chunks=args.n_chunks, n_pool_chunks=args.n_pool_chunks,
                      soft_topm=(None if args.soft_topm <= 0 else args.soft_topm), y_range=y_range, fixed_stats=args.kumo_fixed_stats, self_attention=args.kumo_self_attn, relax=args.kumo_relax)


def add_args(p):
    base.add_args(p)
    p.set_defaults(uq="kumo", out_root=RESULTS_ROOT, objective="gain")
    for a in p._actions:  # allow --uq kumo
        if a.dest == "uq":
            a.choices = ["kumo"]
    p.add_argument("--kumo_size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--kumo_relax", default="weighted", choices=["weighted", "avg_input", "avg_emb"],
                   help="how a relaxed batch enters Kumo: fractional item weights, or b slots averaging raw inputs / row embeddings")
    p.add_argument("--kumo_self_attn", default="weighted", choices=["weighted", "full"],
                   help="ICL self-attention of a soft row: weighted (first grids) or full (a row always sees itself; adaptive_sampling_lib.uq_kumo)")
    p.add_argument("--kumo_fixed_stats", default="none", choices=["none", "target", "features", "both"],
                   help="preprocessing statistics taken from D for the whole epoch instead of each (weighted) context's")
    return p


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    assert args.objective in ("gain", "gain_paired", "risk"), args.objective
    dev = pick_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(args.seed_dataset)
    prob = base.build_problem(args, dev)
    print(prob.summary(), flush=True)
    predictor = train_predictor(prob.train_x, prob.train_y, "regression", seed=42, device=dev)
    psi = predictor_outputs(predictor, prob.eval_x, "regression")
    target = MSETarget(psi, noise_var=0.0)  # Kumo's quantiles are those of the noisy label
    uq = build_uq(args, prob, dev)
    truth = None
    if args.dataset.startswith("synthetic"):  # generating GP known -> denoised target value for the error metric
        truth = oracle_target(prob, psi, prob.meta["lengthscale"], args.outputscale, args.noise_std)
        print("truth:", truth, flush=True)
    seed_all(args.seed)
    dname = args.dataset + (f"_d{args.input_dim}" if args.dataset.startswith("synthetic") else "")
    out_dir = os.path.join(args.out_root, args.tag, f"kumo-{args.kumo_size[0]}_{dname}" + ("_skewed" if args.layout == "skewed" else "")
                           + ("_sf" if args.kumo_self_attn == "full" else "") + {"weighted": "", "avg_input": "_ai", "avg_emb": "_ae"}[args.kumo_relax] + (f"_x{args.eval_scale}" if args.eval_scale != 1 else "") + ("_rng" if args.moments == "range" else "") + (f"_c{args.n_chunks}" if args.n_chunks != 4 else ""))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose, objective=args.objective,
              grad_chunk=args.grad_chunk, imagined_labels=args.imagined_labels)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
