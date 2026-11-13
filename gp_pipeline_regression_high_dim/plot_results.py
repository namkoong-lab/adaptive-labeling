"""Plot / tabulate experiment-1 results.  python plot_results.py [--tag low_compute]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptive_sampling_lib.aggregate import load_runs, plot_two_panel, print_table  # noqa: E402

RESULTS_ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/gp_high_dim"
HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["low_compute", "high_compute"])
p.add_argument("--dims", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
args = p.parse_args()
for tag in args.tags:
    for d in args.dims:
        runs = load_runs(os.path.join(RESULTS_ROOT, tag, f"dim{d}", "*.json"))
        if tag == "high_compute":  # baselines are regime-independent: borrow them from low_compute
            runs += [r for r in load_runs(os.path.join(RESULTS_ROOT, "low_compute", f"dim{d}", "*.json"))
                     if r["method"] in ("random", "uncertainty", "uncertainty_seq")]
        if not runs:
            continue
        print(f"\n===== {tag}  input_dim={d}  ({len(runs)} runs) =====")
        print_table(runs, "var_target", T=10, t_list=[0, 2, 5, 10])
        print_table(runs, "abs_err", T=10, t_list=[0, 2, 5, 10])
        plot_two_panel(runs, f"Synthetic GP, d={d}, {tag.replace('_', '-')}", os.path.join(HERE, "figures", f"{tag}_dim{d}.png"), T=10)
print("figures in", os.path.join(HERE, "figures"))
