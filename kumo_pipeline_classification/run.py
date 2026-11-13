#!/usr/bin/env python
"""Classification (log-loss target) with Kumo-Tabular (NVIDIA; medium by default) as the UQ module (adaptive_sampling_lib.uq_kumo).

Same problems, predictor, target, planners and arguments as tabpfn_pipeline_classification/run.py (whose build_problem and
add_args are reused); the belief is Kumo-Tabular's classifier with the weighted pseudo-update, and, as for regression, the
labelled data's preprocessing statistics in the planner's pseudo-updates (--kumo_fixed_stats both, the default here).
Objectives gain, gain_paired and risk; relaxations --kumo_relax (adaptive_sampling_lib.uq_kumo).  --input_transform and --fp32 do not apply (Kumo's own preprocessing, float32).
Results: results/kumo_classification/<tag>/kumo-<size>_<dataset>[_skewed].

Usage:
  python run.py --dataset magic --layout skewed --method pathwise --tau 0.5 --seed 1 --seed_dataset 1 --n_steps 50 --gpu 0
"""
import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from adaptive_sampling_lib.predictor import train_predictor, predictor_outputs  # noqa: E402
from adaptive_sampling_lib.runner import run_acquisition  # noqa: E402
from adaptive_sampling_lib.targets import LogLossTarget  # noqa: E402
from adaptive_sampling_lib.utils import pick_device, seed_all  # noqa: E402

_spec = importlib.util.spec_from_file_location("tabpfn_cls_run", os.path.join(HERE, "..", "tabpfn_pipeline_classification", "run.py"))
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
RESULTS_ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/kumo_classification"


def build_uq(args, prob, dev):
    from adaptive_sampling_lib.uq_kumo import KumoModule

    return KumoModule(args.kumo_size, device=dev, n_draws=args.n_mc, n_chunks=args.n_chunks, n_pool_chunks=args.n_pool_chunks,
                      soft_topm=(None if args.soft_topm <= 0 else args.soft_topm), fixed_stats=args.kumo_fixed_stats,
                      task="classification", n_classes=2, self_attention=args.kumo_self_attn, relax=args.kumo_relax)


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
    p.add_argument("--kumo_fixed_stats", default="both", choices=["none", "features", "both"],
                   help="preprocessing statistics taken from D for the whole epoch in the pseudo-updates (both = features for classification)")
    return p


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    assert args.objective in ("gain", "gain_paired", "risk"), args.objective
    dev = pick_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(args.seed_dataset)
    prob = base.build_problem(args, dev)
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
    out_dir = os.path.join(args.out_root, args.tag, f"kumo-{args.kumo_size[0]}_{dname}" + ("_skewed" if args.layout == "skewed" else "")
                           + ("_sf" if args.kumo_self_attn == "full" else "") + {"weighted": "", "avg_input": "_ai", "avg_emb": "_ae"}[args.kumo_relax] + (f"_x{args.eval_scale}" if args.eval_scale != 1 else "") + ("" if args.kumo_fixed_stats == "both" else f"_fs{args.kumo_fixed_stats}"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.method}_ds{args.seed_dataset}_s{args.seed}.json")
    pk = dict(n_steps=args.n_steps, G=args.G, n_mc=args.n_mc, lr=args.lr, tau=args.tau, verbose=args.verbose, objective=args.objective,
              grad_chunk=args.grad_chunk, imagined_labels=args.imagined_labels)
    run_acquisition(prob, uq, target, args.method, args.b, args.T, args.seed, out_path, planner_kwargs=pk,
                    n_track=args.n_track, verbose=True, extra_record={"args": vars(args), "device": str(dev)}, truth=truth)
    print("saved", out_path)


if __name__ == "__main__":
    main()
