"""Tables and two-panel figures (posterior variance of the target; |error| of its estimate) of GP runs, per tag and problem.

  python plot_results.py   (the paper grid: low and high compute, pathwise at tau = 0.5)
  python plot_results.py --dataset eicu --tags eicu_low_compute eicu_high_compute
  python plot_results.py --dims 1 --tags rollouts_G1_high_compute rollouts_G2_high_compute \
      draws_high_compute:reinforce,reinforce_baseline+draws_tau05_high_compute --baseline_tag draws_low_compute

A tag "a+b" merges the runs of several tags into one figure; "tag:m1,m2" keeps only those methods of a tag.  Baselines are
run once (low compute) and are added to every figure from --baseline_tag (default: the first tag with its regime replaced
by 'low', e.g. eicu_high_compute -> eicu_low_compute).  The error is against the generating GP's denoised target
(synthetic) or the evaluation labels (eicu).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from adaptive_sampling_lib.aggregate import load_runs, plot_two_panel, print_table  # noqa: E402

from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("gp_high_dim")
BASELINES = ("random", "uncertainty", "uncertainty_seq")

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="synthetic", choices=["synthetic", "eicu"])
p.add_argument("--tags", nargs="+", default=["draws_low_compute:random,uncertainty,uncertainty_seq,reinforce,reinforce_baseline+draws_tau05_low_compute",
                                             "draws_high_compute:reinforce,reinforce_baseline+draws_tau05_high_compute"])
p.add_argument("--baseline_tag", default=None)
p.add_argument("--dims", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100], help="synthetic only")
p.add_argument("--T", type=int, default=10)
p.add_argument("--out", default=results("analysis", "gp_high_dim", "figures"))
args = p.parse_args()
err = "abs_err_oracle" if args.dataset == "synthetic" else "abs_err"
subs = [f"dim{d}" for d in args.dims] if args.dataset == "synthetic" else [args.dataset]


def tag_runs(item, sub):
    tag, _, keep = item.partition(":")
    runs = load_runs(os.path.join(RESULTS_ROOT, tag, sub, "*.json"))
    return [r for r in runs if not keep or r["method"] in keep.split(",")]


for spec in args.tags:
    tags = [t.partition(":")[0] for t in spec.split("+")]
    btag = args.baseline_tag or re.sub(r"(high|mid\d+)_compute$", "low_compute", tags[0])
    for sub in subs:
        runs = [r for item in spec.split("+") for r in tag_runs(item, sub)]
        if btag not in tags:  # baselines do not depend on the compute regime
            runs += [r for r in load_runs(os.path.join(RESULTS_ROOT, btag, sub, "*.json")) if r["method"] in BASELINES]
        if not runs:
            continue
        print(f"\n===== {spec}  {sub}  ({len(runs)} runs) =====")
        print_table(runs, "var_target", T=args.T, t_list=[0, 2, 5, args.T])
        print_table(runs, err, T=args.T, t_list=[0, 2, 5, args.T])
        name = "_".join(tags) + f"_{sub}.png"
        plot_two_panel(runs, f"GP, {sub}, {' + '.join(tags)}", os.path.join(args.out, name), T=args.T, keys=("var_target", err))
print("figures in", args.out)
