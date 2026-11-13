"""The paper's remaining GP figures, in the style of make_figures.py:

  gp_eicu_{low,high}_compute.pdf  GP on eICU, all six methods, 1 / 1000 gradient steps per epoch, ten training seeds
                                  (experiments/gp/make_jobs.py --grid eicu)
  gp_rollouts_vs_N.pdf            planning with N = 1, 2, 10 rollouts per gradient estimate, synthetic GP (d = 1), 100
                                  steps per epoch, twenty function draws (make_jobs.py --grid rollouts; N = 10 is the
                                  paper grid's high-compute run)
  cosine_similarity_vs_N.pdf      expected cosine similarity between each gradient estimator and the true gradient against
                                  the number of rollouts N (experiments/gp/gradient_quality.py); mean over the function
                                  draws found, band = one standard error across draws (within a draw if there is one)

Arrows in the variance panels mark label savings of the pathwise planner, as in make_figures.py.

  python paper_figures/make_gp_extra_figures.py [--out <folder>] [--taus 0.1 0.5 10]
"""
import argparse
import glob
import json
import os

import numpy as np
from matplotlib.lines import Line2D

import make_figures as mrf
from make_figures import COLOR, INK2, METHODS, BASELINES, T, plt, plot_curve, style_axis, add_legend, save

GP = os.path.join(mrf.RESULTS, "gp_high_dim")
VAR_LABEL, RMSE_LABEL = "Var of $\\ell_2$ loss", "RMSE of estimated model $\\ell_2$ loss"


def two_panel_layout():
    return plt.subplots(1, 2, figsize=(15.5, 6.6))


def fig_eicu():
    """GP on eICU, one figure per compute regime, all six methods (layout of gp_d1_{low,high}_compute.pdf)."""
    for regime, steps in (("low", 1), ("high", 1000)):
        cell = {}
        for m in METHODS:
            tag = "eicu_low_compute" if m in BASELINES else f"eicu_{regime}_compute"
            cell[m] = mrf.load_group(os.path.join(GP, tag, "eicu"), m, "real")
        ks = mrf.common_splits(cell)
        if not ks:
            print(f"eICU {regime}: no complete set of runs, skipped")
            continue
        cell = {m: {k: v[k] for k in ks} for m, v in cell.items()}
        print(f"GP eICU {regime} compute ({steps} steps): {len(ks)} training seeds")
        fig, axes = two_panel_layout()
        for m in METHODS:
            vm, vse = mrf.mean_se(cell[m], 0)
            em, ese = mrf.rmse_se(cell[m])
            plot_curve(axes[0], vm, vse, vse, m)
            plot_curve(axes[1], em, ese, ese, m)
        mrf.mark_savings(axes[0], mrf.mean_se(cell["pathwise"], 0)[0], {m: mrf.mean_se(cell[m], 0)[0] for m in mrf.SAVINGS_REFS})
        style_axis(axes[0], VAR_LABEL)
        style_axis(axes[1], RMSE_LABEL)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        add_legend(fig)
        mrf.draw_savings(fig)
        save(fig, f"gp_eicu_{regime}_compute.pdf")


def fig_rollouts():
    """Planning with N = 1, 2, 10 rollouts per gradient estimate: colour = method, line style = N."""
    methods = ["pathwise", "reinforce_baseline", "reinforce"]
    folder = {1: "rollouts_G1_high_compute", 2: "rollouts_G2_high_compute"}
    cell = {}
    for m in methods:
        for n in (1, 2, 10):
            tag = folder.get(n) or ("draws_tau05_high_compute" if m == "pathwise" else "draws_high_compute")
            cell[(m, n)] = mrf.load_group(os.path.join(GP, tag, "dim1"), m, "oracle")
    ks = mrf.common_splits(cell)
    if not ks:
        print("rollouts: no complete set of runs, skipped")
        return
    cell = {key: {k: v[k] for k in ks} for key, v in cell.items()}
    print(f"GP rollouts vs N: {len(ks)} function draws")
    name = {"pathwise": "AUTODIFF", "reinforce_baseline": "REINFORCE (with baseline)", "reinforce": "REINFORCE (without baseline)"}
    ls_of_n = {1: "-", 2: (0, (5, 2)), 10: (0, (1, 1.6))}
    fig, axes = two_panel_layout()
    for m in ["reinforce", "reinforce_baseline", "pathwise"]:  # REINFORCE without baseline first: at N = 1 the two coincide
        for n in (1, 2, 10):
            for ax, (mean, se) in zip(axes, (mrf.mean_se(cell[(m, n)], 0), mrf.rmse_se(cell[(m, n)]))):
                x = np.arange(len(mean))
                ax.fill_between(x, mean - se, mean + se, color=COLOR[m], alpha=0.06, lw=0, zorder=2)
                ax.plot(x, mean, color=COLOR[m], ls=ls_of_n[n], lw=3.2, zorder=3, solid_capstyle="round", dash_capstyle="round")
    handles, labels = [], []
    for m in methods:  # legend columns = methods, rows = N
        for n in (1, 2, 10):
            handles.append(Line2D([], [], color=COLOR[m], ls=ls_of_n[n], lw=3.4, dash_capstyle="round"))
            labels.append(f"{name[m]} ($N={n}$)")
    var = lambda m, n: mrf.mean_se(cell[(m, n)], 0)[0]
    mrf.mark_savings(axes[0], var("pathwise", 1), {"reinforce": var("reinforce", 10), "reinforce_baseline": var("reinforce_baseline", 10)})
    style_axis(axes[0], VAR_LABEL)
    style_axis(axes[1], RMSE_LABEL)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=17, bbox_to_anchor=(0.5, 0.0), frameon=False,
               handlelength=3.0, columnspacing=1.8, labelcolor=INK2)
    mrf.draw_savings(fig)
    save(fig, "gp_rollouts_vs_N.pdf")


def fig_cosine(taus, files):
    """Cosine similarity to the true gradient against the number of rollouts N (equally spaced, labelled by N).

    Drawn at its printed size (inches, font sizes in pt) and included at natural size: the plot with its axis labels takes
    the left 3.25 in (half of the 6.5 in text width) and the legend the remaining 1.7 in beside it."""
    if not files:
        print("gradient quality: no result files, skipped")
        return
    ds = [json.load(open(f)) for f in files]
    Ns = ds[0]["Ns"]
    print(f"gradient quality: {len(ds)} function draws ({', '.join(os.path.basename(f) for f in files)})")

    def stat(key, n):
        v = [d["cos"][f"{key}@N{n}"] for d in ds]
        m = np.mean([x["mean"] for x in v])
        se = np.std([x["mean"] for x in v], ddof=1) / np.sqrt(len(v)) if len(v) > 1 else v[0]["se"]
        return m, se

    # The two families differ in more than colour: AUTODIFF = solid lines with filled markers in three greens (light to
    # dark with tau), REINFORCE = dashed lines with open markers, in its colours from the other figures.
    greens, gmarks = ["#7FCFB2", COLOR["pathwise"], "#00684B"], ["s", "o", "^"]
    series = [(f"pathwise_tau{float(t)}", f"AUTODIFF ($\\tau = {t:g}$)", g, "-", mk, True)
              for t, g, mk in zip(sorted(taus), greens, gmarks)]
    series += [("reinforce", "REINFORCE\n(without baseline)", COLOR["reinforce"], (0, (3, 1.3, 1, 1.3)), "v", False),
               ("reinforce_baseline", "REINFORCE\n(with baseline)", COLOR["reinforce_baseline"], (0, (3.5, 1.6)), "D", False)]
    W, H, PLOT = 4.95, 2.4, 3.25          # figure width, height, width of the plot part (inches)
    left, right, bottom, top = 0.6, 0.06, 0.45, 0.08  # margins of the axes inside the plot part (inches)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, (PLOT - left - right) / W, (H - bottom - top) / H])
    x = np.arange(len(Ns))
    handles, lo = [], 0.0
    for key, label, color, ls, marker, filled in series:
        m, se = map(np.array, zip(*[stat(key, n) for n in Ns]))
        lo = min(lo, float((m - se).min()))
        main = key == "pathwise_tau0.5"
        marks = dict(mfc=color, mec="white", mew=0.5, ms=5.5) if filled else dict(mfc="white", mec=color, mew=1.1, ms=5.0)
        style = dict(color=color, ls=ls, lw=1.8 if main else 1.5, marker=marker, solid_capstyle="round", dash_capstyle="butt", **marks)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, m, zorder=4 if main else 3, **style)
        handles.append(Line2D([], [], **style))
    style_axis(ax, "Cosine similarity", xlabel="Rollouts per estimate $N$", fs=9.5, ts=8.5)
    ax.yaxis.labelpad = 4
    ax.tick_params(length=2.5, width=0.5, pad=2)
    ax.grid(True, axis="y", color=mrf.GRID, lw=0.45)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    if lo < 0:
        ax.axhline(0, color=mrf.SPINE, lw=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_yticks(np.arange(np.floor(lo * 5) / 5, 1.01, 0.2))
    ax.set_xlim(-0.3, len(Ns) - 0.7)
    ax.set_ylim(min(-0.03, lo - 0.03), 1.02)
    # the legend fills the space right of the plot part, one column, vertically centred on the axes
    fig.legend(handles, [s[1] for s in series], loc="center left", fontsize=8.5, frameon=False,
               bbox_to_anchor=((PLOT + 0.12) / W, (bottom + (H - bottom - top) / 2) / H),
               handlelength=2.6, labelspacing=0.7, borderaxespad=0, labelcolor=INK2)
    os.makedirs(mrf.OUT, exist_ok=True)
    fig.savefig(os.path.join(mrf.OUT, "cosine_similarity_vs_N.pdf"))  # fixed size: no tight bounding box
    plt.close(fig)
    print("saved cosine_similarity_vs_N.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=mrf.OUT)
    p.add_argument("--taus", type=float, nargs=3, default=[0.1, 0.5, 10.0], help="the three pathwise temperatures of the cosine figure")
    p.add_argument("--gradient_files", nargs="+", default=None,
                   help="gradient-quality JSONs (default: every synthetic d = 1 draw under <results>/gp_high_dim/gradient_quality)")
    a = p.parse_args()
    mrf.OUT = a.out
    fig_eicu()
    fig_rollouts()
    fig_cosine(a.taus, a.gradient_files or sorted(glob.glob(os.path.join(GP, "gradient_quality", "gradient_quality_dim1_ds*.json"))))


if __name__ == "__main__":
    main()
