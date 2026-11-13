"""Agreement between posterior variance and realised squared error across independent dataset splits (TabPFN).

Mirrors experiments/gp/analyze_draws.py for the eICU-regression split sweep (tags skew_splits_low/high,
group tabpfn_eicu_raw_skewed, one run per dataset seed k with training seed k).  eICU regression labels are noise-free
random-forest probabilities, so the realised target equals the population target at the evaluation inputs and no oracle
correction is needed.

  python analyze_splits.py [--T 10] [--tag_prefix skew_splits] [--group tabpfn_eicu_raw_skewed]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from adaptive_sampling_lib.paths import results  # noqa: E402
R = results("tabpfn_regression")
METHODS = ["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"]
SHORT = {"random": "Random", "uncertainty": "Uncertainty (static)", "uncertainty_seq": "Uncertainty (sequential)",
         "reinforce": "REINFORCE (no baseline)", "reinforce_baseline": "REINFORCE (mean baseline)", "pathwise": "Pathwise (Autodiff)"}
p = argparse.ArgumentParser()
p.add_argument("--T", type=int, default=10)
p.add_argument("--tag_prefix", default="skew_splits")
p.add_argument("--group", default="tabpfn_eicu_raw_skewed")
p.add_argument("--out", default=results("analysis", "tab_splits.tex"))
p.add_argument("--root", default=None, help="results root (default: tabpfn_regression)")
p.add_argument("--oracle", action="store_true", help="error against actual_target_oracle (synthetic data) instead of the realised target")
p.add_argument("--n_steps_high", type=int, default=50)
a = p.parse_args()
if a.root:
    R = a.root


def load(tag, m):
    out = []
    for f in sorted(glob.glob(f"{R}/{tag}/{a.group}/{m}_ds*_s*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = r["horizon"]
        if len(h) < a.T + 1:
            continue
        x = h[a.T]
        truth = x["actual_target_oracle"] if a.oracle else x["actual_target"]
        out.append((x["var_target"], (x["est_target"] - truth) ** 2, r["args"]["seed_dataset"]))
    return out


print(f"{'regime':5s} {'method':26s} {'n':>3s} {'sqrt(mean Var)':>14s} {'RMSE':>8s} {'mean z^2':>12s}  per-split z")
rows = {}
for reg in ("low", "high"):
    for m in METHODS:
        tag = f"{a.tag_prefix}_low" if (m in METHODS[:3] or reg == "low") else f"{a.tag_prefix}_high"
        v = load(tag, m)
        if not v:
            continue
        var = np.array([x[0] for x in v]); se2 = np.array([x[1] for x in v])
        z = np.sign(1) * np.sqrt(se2 / var)
        z2 = se2 / var
        rows[(reg, m)] = (len(v), np.sqrt(var.mean()), np.sqrt(se2.mean()), z2.mean(), z2.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        print(f"{reg:5s} {SHORT[m]:26s} {len(v):3d} {np.sqrt(var.mean()):14.4g} {np.sqrt(se2.mean()):8.4g} {z2.mean():6.2f}+-{rows[(reg, m)][4]:4.2f}  " + " ".join(f"{s:4.2f}" for s in np.sqrt(z2)))
lines = []
for reg, label in (("low", "Low compute (1 step / epoch)"), ("high", f"High compute ({a.n_steps_high} steps / epoch)")):
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}}\\\\")
    for m in METHODS:
        if (reg, m) not in rows:
            continue
        n, sv, rm, z2m, z2se = rows[(reg, m)]
        lines.append(f"{SHORT[m]} & {sv:.3g} & {rm:.3g} & {z2m:.2f}$\\pm${z2se:.2f}\\\\")
    lines.append("\\midrule")
lines = lines[:-1]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\nMethod & $\\sqrt{\\overline{\\mathrm{Var}}}$ & RMSE & $\\overline{z^2}$\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote", a.out)
