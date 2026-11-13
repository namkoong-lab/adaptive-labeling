"""Generic result summariser / plotter for any experiment folder.

  python -m adaptive_sampling_lib.plot_runs --root <results_root> --tag pilot_high [--baseline_tag pilot_low]
         --group tabpfn_eicu --out figures/ [--T 10] [--keys var_target abs_err]

For every <root>/<tag>/<group>/ it prints the mean +- s.e. table at a few horizons and writes a
two-panel figure; baselines (random / uncertainty) are regime independent and are borrowed from
--baseline_tag when they are missing from --tag.
"""
import argparse
import glob
import os

from .aggregate import load_runs, plot_two_panel, print_table

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--baseline_tag", default=None)
p.add_argument("--groups", nargs="*", default=None, help="sub-folders (datasets); default: all")
p.add_argument("--out", default=None)
p.add_argument("--T", type=int, default=None)
p.add_argument("--keys", nargs="+", default=["var_target", "abs_err"])
p.add_argument("--logy", action="store_true")
p.add_argument("--t_list", type=int, nargs="+", default=None)
a = p.parse_args()
for tag in a.tags:
    groups = a.groups or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.root, tag, "*")) if os.path.isdir(d))
    for g in groups:
        runs = load_runs(os.path.join(a.root, tag, g, "*.json"))
        if a.baseline_tag and a.baseline_tag != tag:
            have = {r["method"] for r in runs}
            runs += [r for r in load_runs(os.path.join(a.root, a.baseline_tag, g, "*.json"))
                     if r["method"] in ("random", "uncertainty", "uncertainty_seq") and r["method"] not in have]
        if not runs:
            continue
        T = a.T or max(len(r["horizon"]) for r in runs) - 1
        print(f"\n===== {tag} / {g}  ({len(runs)} runs, T={T}) =====")
        for k in a.keys:
            print_table(runs, k, T=T, t_list=a.t_list or sorted({0, T // 4, T // 2, T}), fmt="{:.3g}")
        if a.out:
            ylab = ("Var of target", "|estimate - realised target|")
            plot_two_panel(runs, f"{g} ({tag})", os.path.join(a.out, f"{tag}_{g}.png"), T=T, keys=tuple(a.keys[:2]),
                           ylabels=ylab, logy=(a.logy, a.logy))
