"""Aggregate run JSONs into per-method curves (mean +- s.e. over seeds) and plot the paper-style
two-panel figure (posterior variance of the target; |estimate - realised target|) vs. horizon."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

LABELS = {
    "random": "Random sampling",
    "uncertainty": "Uncertainty sampling - static",
    "uncertainty_seq": "Uncertainty sampling - sequential",
    "reinforce": "REINFORCE (without baseline) 1-lookahead",
    "reinforce_baseline": "REINFORCE (with baseline) 1-lookahead",
    "reinforce_loo": "REINFORCE (LOO baseline) 1-lookahead",
    "pathwise": "AUTODIFF 1-lookahead",
}
COLORS = {
    "random": "tab:blue", "uncertainty": "tab:green", "uncertainty_seq": "tab:purple",
    "reinforce": "tab:red", "reinforce_baseline": "tab:brown", "reinforce_loo": "tab:pink", "pathwise": "tab:orange",
}
ORDER = ["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "reinforce_loo", "pathwise"]


def load_runs(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except Exception:
            continue
        if "horizon" in r:
            runs.append(r)
    return runs


def _series(r, key):
    """per-epoch values of `key`; squared-error keys are derived on the fly for older result files."""
    h = r["horizon"]
    if key in h[0]:
        return [m[key] for m in h]
    if key == "sq_err":
        return [(m["est_target"] - m["actual_target"]) ** 2 for m in h]
    if key == "sq_err_oracle" and "actual_target_oracle" in h[0]:
        return [(m["est_target"] - m["actual_target_oracle"]) ** 2 for m in h]
    if key == "abs_err_oracle" and "actual_target_oracle" in h[0]:
        return [abs(m["est_target"] - m["actual_target_oracle"]) for m in h]
    return None


def curves(runs, key, T=None, rms=False):
    """-> {method: (mean[T+1], se[T+1], n)} using only complete runs.
    rms=True: root of the mean of `key` over seeds (e.g. RMSE from a squared-error key), s.e. by the delta method."""
    by = defaultdict(list)
    for r in runs:
        h = r["horizon"]
        if T is not None and len(h) < T + 1:
            continue
        v = _series(r, key)
        if v is None:
            continue
        by[r["method"]].append(v[: (T + 1 if T else None)])
    out = {}
    for meth, lst in by.items():
        L = min(len(x) for x in lst)
        a = np.array([x[:L] for x in lst], dtype=float)
        mean = a.mean(0)
        se = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(L)
        if rms:
            root = np.sqrt(np.maximum(mean, 0))
            se = se / np.maximum(2 * root, 1e-300)
            mean = root
        out[meth] = (mean, se, len(a))
    return out


def calibration_ratio(runs, T=None, key_sq="sq_err", key_var="var_target"):
    """-> {method: (ratio[T+1], n)} with ratio_t = mean_seeds(squared error_t) / mean_seeds(posterior variance_t).
    ~1 for a calibrated belief, >>1 for an over-confident one (error larger than the belief's own spread)."""
    num, den = curves(runs, key_sq, T), curves(runs, key_var, T)
    return {m: (num[m][0] / np.maximum(den[m][0], 1e-300), num[m][2]) for m in num if m in den}


def plot_two_panel(runs, title, out_png, T=None, keys=("var_target", "abs_err"),
                   ylabels=("Var of $\\ell_2$ loss", "Error of estimated model $\\ell_2$ loss"), logy=(False, False),
                   methods=None, rms=(False, False)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.8))
    for ax, key, yl, lg, rm in zip(axes, keys, ylabels, logy, rms):
        cv = curves(runs, key, T, rms=rm)
        for meth in ORDER:
            if meth not in cv or (methods is not None and meth not in methods):
                continue
            mean, se, n = cv[meth]
            x = np.arange(len(mean))
            ax.plot(x, mean, marker="o", ms=3, color=COLORS.get(meth), label=f"{LABELS.get(meth, meth)} (n={n})")
            ax.fill_between(x, mean - se, mean + se, color=COLORS.get(meth), alpha=0.15)
        ax.set_xlabel("Time horizon")
        ax.set_ylabel(yl)
        if lg:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def table(runs, key="var_target", T=None, t_list=None, rms=False):
    cv = curves(runs, key, T, rms=rms)
    rows = []
    for meth in ORDER:
        if meth not in cv:
            continue
        mean, se, n = cv[meth]
        ts = t_list or [0, len(mean) // 2, len(mean) - 1]
        rows.append((meth, n, [(int(t), float(mean[t]), float(se[t])) for t in ts if t < len(mean)]))
    return rows


def print_table(runs, key="var_target", T=None, t_list=None, fmt="{:.4g}", rms=False):
    rows = table(runs, key, T, t_list, rms=rms)
    print(f"--- {'RMS of ' if rms else ''}{key} (mean ± se over seeds) ---")
    for meth, n, vals in rows:
        s = "  ".join(f"t={t}: {fmt.format(m)}±{fmt.format(se)}" for t, m, se in vals)
        print(f"{LABELS.get(meth, meth):45s} n={n:2d}  {s}")
