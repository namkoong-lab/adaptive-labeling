"""Figures and tables of the paper's experiments: GP in R^d and Kumo-Tabular (make_gp_extra_figures.py: GP on eICU, the
rollouts-per-estimate experiment and the gradient-quality study).

The plots keep the layout and line styles of the manuscript's earlier figures:
posterior variance of the target on
the left and the error of its estimate on the right.  Style: Okabe-Ito colours
(pathwise planner green), serif (Computer Modern) text, open axes with a light
horizontal grid, a shaded band of one standard error around each mean curve,
thick lines with the pathwise planner thickest, and one frameless legend
(planners in the first row, baselines in the second).

Inputs: the result JSONs written by the experiments (one file per run, with the
per-epoch horizon of the posterior variance and of the estimate), under the
results root of adaptive_sampling_lib/paths.py ($ADAPTIVE_SAMPLING_RESULTS).

Experiments shown
-----------------
* GP regression in R^d, d in {1, 5, 10, 20, 50, 100}, twenty GP function draws
  per dimension.  Low compute = 1 policy-gradient step per decision epoch,
  high compute = 100 steps; pathwise planner at temperature tau = 0.5.
* Kumo-Tabular (medium) as the UQ module on four real regression datasets
  (eICU, California housing, Bike sharing, MIMIC-IV ICU length of stay) and four
  real classification datasets (eICU mortality, MAGIC gamma telescope, credit
  card default, MIMIC-IV mortality), ten splits each.  Low compute = 1 step,
  high compute = 50 steps; pathwise planner at tau = 0.5.

Outputs (written to --out, default paper_figures/output/)
-------------------------------------
  gp_d1_combined.pdf                    main text, d = 1, twenty draws (low and high compute)
  gp_d1_{low,high}_compute.pdf          the same, one regime per figure
  gp_dims_{low,high}_compute.pdf        appendix, every dimension
  kumo_pooled_{low,high}_compute.pdf    pooled over datasets, both tasks (low: main text, high: appendix)
  kumo_{regression,classification}_pooled_{low,high}_compute.pdf   the same, one task per figure
  kumo_{regression,classification}_{low,high}_compute.pdf  appendix, per dataset
  tables/tab_gp_dims.tex                appendix table (all dimensions)
  tables/tab_kumo_{baselines,planners_1,planners_50}.tex   main-text tables (both tasks)
  summary.txt                           numbers quoted in the text

Usage:
  python paper_figures/make_figures.py [--out <folder>]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # repo root
from adaptive_sampling_lib.paths import RESULTS_ROOT as RESULTS  # noqa: E402
OUT = os.path.join(HERE, "output")  # set by --out
T = 10  # decision epochs

# ----------------------------------------------------------------------------
# Plot style: Okabe-Ito colours (distinguishable with colour-vision deficiency; the theory figures use the same
# palette), the pathwise planner in green as in the original figures, and the original line styles
# ----------------------------------------------------------------------------
METHODS = ["random", "pathwise", "uncertainty", "reinforce", "uncertainty_seq", "reinforce_baseline"]  # legend order
BASELINES = ["random", "uncertainty", "uncertainty_seq"]
PLANNERS = ["reinforce", "reinforce_baseline", "pathwise"]
LABEL = {
    "random": "Random sampling",
    "pathwise": "AUTODIFF 1-lookahead",
    "uncertainty": "Uncertainty sampling - static",
    "reinforce": "REINFORCE (without baseline) 1-lookahead",
    "uncertainty_seq": "Uncertainty sampling - sequential",
    "reinforce_baseline": "REINFORCE (with baseline) 1-lookahead",
}
COLOR = {
    "random": "#E69F00",              # orange
    "pathwise": "#009E73",            # bluish green
    "uncertainty": "#CC79A7",         # reddish purple
    "reinforce": "#4D4D4D",           # dark grey
    "uncertainty_seq": "#D55E00",     # vermillion
    "reinforce_baseline": "#0072B2",  # blue
}
STYLE = {
    "random": ":", "pathwise": "-", "uncertainty": (0, (8, 2, 1.5, 2)),
    "reinforce": (0, (3, 1.5, 1, 1.5, 1, 1.5)), "uncertainty_seq": (0, (10, 2, 1.5, 2)),
    "reinforce_baseline": "--",
}
SHORT = {"uncertainty": "U", "uncertainty_seq": "U-seq", "reinforce": "R", "reinforce_baseline": "Rb", "pathwise": "PW"}
# Legend order: ncol=3 fills column by column, so the first row holds the planners and the second the baselines.
LEGEND_ORDER = ["pathwise", "random", "reinforce_baseline", "uncertainty", "reinforce", "uncertainty_seq"]
XLABEL = "Decision epoch $t$"
INK, INK2, MUTED, SPINE, GRID = "#1a1a1a", "#4a4a4a", "#8a8a8a", "#b5b4ad", "#e4e3dd"  # text, axes and grid greys
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})


def plot_curve(ax, mean, lo, hi, method, lw=2.4):
    """One method's mean curve with a shaded band of one standard error (lo, hi are distances below / above the
    mean).  The pathwise planner is drawn thicker, on top, with a stronger band."""
    x = np.arange(len(mean))
    pw = method == "pathwise"
    ax.fill_between(x, mean - lo, mean + hi, color=COLOR[method], alpha=0.16 if pw else 0.05, lw=0, zorder=2)
    ax.plot(x, mean, color=COLOR[method], ls=STYLE[method], lw=lw * (1.75 if pw else 1.25), zorder=4 if pw else 3,
            label=LABEL[method], solid_capstyle="round", dash_capstyle="round")


def style_axis(ax, ylabel=None, xlabel=XLABEL, fs=20, ts=15):
    """Open axes (no top / right spine), light horizontal grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.9)
    ax.grid(True, axis="y", color=GRID, lw=0.9, ls="-")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, T + 1, 2))
    ax.tick_params(labelsize=ts, colors=MUTED, labelcolor=INK2, length=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs, color=INK2, labelpad=8)


def set_title(ax, text, fs=19):
    """Panel title: regular weight, left-aligned, in ink."""
    ax.set_title(text, fontsize=fs, color=INK, loc="left", pad=10)


def add_legend(fig, fs=17, y=0.0):
    """One frameless legend under the panels, three columns: the first row lists the planners, the second the
    baselines.  The legend uses its own handles (slightly thicker lines) so the dotted line stays visible."""
    present = {l for ax in fig.axes for l in ax.get_legend_handles_labels()[1]}
    keys = [k for k in LEGEND_ORDER if LABEL[k] in present]
    handles = [Line2D([], [], color=COLOR[k], ls=STYLE[k], lw=4.4 if k == "pathwise" else 3.4,
                      solid_capstyle="round", dash_capstyle="round") for k in keys]
    fig.legend(handles, [LABEL[k] for k in keys], loc="lower center", ncol=3, fontsize=fs, bbox_to_anchor=(0.5, y),
               frameon=False, handlelength=3.0, columnspacing=1.8, labelcolor=INK2)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_run(path, err_key):
    """-> (posterior variance [T+1], squared error of the estimate [T+1]) or None for an unfinished run."""
    try:
        r = json.load(open(path))
    except Exception:
        return None
    h = r["horizon"]
    if len(h) < T + 1:
        return None
    var = np.array([h[t]["var_target"] for t in range(T + 1)])
    if err_key == "oracle":  # synthetic GP: error against the exact (oracle) target
        err = np.array([(h[t]["est_target"] - h[t]["actual_target_oracle"]) ** 2 for t in range(T + 1)])
    else:  # real data: error against the realised target on the evaluation set
        err = np.array([(h[t]["est_target"] - h[t]["actual_target"]) ** 2 for t in range(T + 1)])
    return var, err, r.get("timing")


def load_group(folder, method, err_key):
    """{split id: (var, err, timing)} for every finished run of `method` in `folder`."""
    out = {}
    for f in glob.glob(os.path.join(folder, f"{method}_ds*_s*.json")):
        run = read_run(f, err_key)
        if run is not None:
            out[os.path.basename(f)[len(method) + 1:-5]] = run
    return out


def common_splits(cell):
    keys = set.intersection(*[set(v) for v in cell.values()])
    return sorted(keys, key=lambda k: int(k.split("_")[0][2:]))


# GP: (method, steps) -> folder per dimension
GP_DIMS = [1, 5, 10, 20, 50, 100]
GP_BUDGETS = [("low", 1), ("high", 100)]


def gp_folder(method, regime, d):
    if method == "pathwise":
        tag = f"draws_tau05_{regime}_compute"  # tau = 0.5
    elif method in BASELINES:
        tag = "draws_low_compute"  # baselines do no policy optimisation
    else:
        tag = f"draws_{regime}_compute"
    return os.path.join(RESULTS, "gp_high_dim", tag, f"dim{d}")


def load_gp():
    data = {}
    for d in GP_DIMS:
        cell = {}
        for regime, steps in GP_BUDGETS:
            for m in METHODS:
                key = (m, 0) if m in BASELINES else (m, steps)
                if key not in cell:
                    cell[key] = load_group(gp_folder(m, regime, d), m, "oracle")
        ks = common_splits(cell)
        data[d] = {k: {s: v[s] for s in ks} for k, v in cell.items()}
        print(f"GP d={d}: {len(ks)} function draws")
    return data


KUMO = {
    "regression": dict(root=os.path.join(RESULTS, "kumo_regression"), tag="final_kumo_sf", suffix="_skewed_sf_rng",
                       sets=[("eicu_raw", "eICU"), ("california", "California"), ("bike", "Bike"), ("mimic_los", "MIMIC-IV LOS")],
                       target="$\\ell_2$ loss"),
    "classification": dict(root=os.path.join(RESULTS, "kumo_classification"), tag="final_kumo_cls_sf", suffix="_skewed_sf",
                           sets=[("eicu", "eICU"), ("magic", "MAGIC"), ("credit", "Credit"), ("mimic", "MIMIC-IV mortality")],
                           target="log-loss"),
}
KUMO_BUDGETS = [("low", 1), ("high", 50)]


def load_kumo(task):
    spec = KUMO[task]
    data = {}
    for ds, lab in spec["sets"]:
        cell = {}
        for regime, steps in KUMO_BUDGETS:
            folder = os.path.join(spec["root"], f"{spec['tag']}_{regime}", f"kumo-m_{ds}{spec['suffix']}")
            for m in METHODS:
                key = (m, 0) if m in BASELINES else (m, steps)
                if key not in cell:
                    cell[key] = load_group(folder, m, "real")
        ks = common_splits(cell)
        data[lab] = {k: {s: v[s] for s in ks} for k, v in cell.items()}
        print(f"Kumo {task} {lab}: {len(ks)} splits")
    return data


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def mean_se(runs, idx):
    """Mean +- s.e. over runs of the series runs[s][idx] ([T+1])."""
    a = np.array([r[idx] for r in runs.values()])
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


def rmse_se(runs):
    """RMSE over runs (root of the mean squared error) with a delta-method s.e."""
    m, se = mean_se(runs, 1)
    root = np.sqrt(m)
    return root, se / np.maximum(2 * root, 1e-300)


def rel_var_curve(datasets, key):
    """Posterior variance relative to t=0, geometric mean over all splits of all datasets (+- one s.e. of the log)."""
    logs = []
    for cell in datasets:
        for s, run in cell[key].items():
            logs.append(np.log(run[0] / run[0][0]))
    logs = np.array(logs)
    m, se = logs.mean(0), logs.std(0, ddof=1) / np.sqrt(len(logs))
    g = np.exp(m)
    return g, g - np.exp(m - se), np.exp(m + se) - g


def rel_rmse_curve(datasets, key):
    """RMSE relative to t=0 per dataset, geometric mean over datasets (+- delta-method s.e. of the log)."""
    logs, var_logs = [], []
    for cell in datasets:
        m, se = mean_se(cell[key], 1)
        logs.append(0.5 * np.log(m / m[0]))
        var_logs.append((se / (2 * np.maximum(m, 1e-300))) ** 2)
    m = np.mean(logs, 0)
    se = np.sqrt(np.sum(var_logs, 0)) / len(logs)
    g = np.exp(m)
    return g, g - np.exp(m - se), np.exp(m + se) - g


def vs_random(cell, key):
    """Final (t=T) variance: mean log ratio to random on the same split; RMSE ratio to random; per-split log ratios."""
    ref, x = cell[("random", 0)], cell[key]
    ks = sorted(set(ref) & set(x))
    lr = np.log([x[k][0][T] / ref[k][0][T] for k in ks])
    rr = np.sqrt(np.mean([x[k][1][T] for k in ks]) / np.mean([ref[k][1][T] for k in ks]))
    return lr, rr


def paired(cell, a, b):
    """Final variance of a vs b on the same splits: mean log ratio, s.e., wins of a, n."""
    ks = sorted(set(cell[a]) & set(cell[b]))
    lr = np.log([cell[a][k][0][T] / cell[b][k][0][T] for k in ks])
    return lr.mean(), lr.std(ddof=1) / np.sqrt(len(lr)), int((lr < 0).sum()), len(lr)


pct = lambda lr: 100.0 * (np.exp(lr) - 1.0)


def batches_to_reach(curve, target):
    """First decision epoch t >= 1 at which `curve` (posterior variance over the horizon) is at or below `target`;
    None if it never is.  Labels come in batches, so t is an integer number of batches."""
    for t in range(1, T + 1):
        if curve[t] <= target * (1 + 1e-12):
            return t
    return None


def savings_final(pw, ref):
    """Label savings of the planner (curve `pw`) against a reference method (curve `ref`) at the reference's final
    stage: 10 / (batches the planner needs to reach ref's final variance).  If the planner never gets there, the
    factor is (batches the reference needs to reach the planner's final variance) / 10, which is below 1."""
    tp = batches_to_reach(pw, ref[T])
    if tp:
        return T / tp
    tb = batches_to_reach(ref, pw[T])
    return tb / T if tb else 1.0


def savings_best(pw, ref):
    """Most favourable stage: max over the reference's stages t of t / (batches the planner needs to reach ref[t])."""
    best = [t / tp for t in range(1, T + 1) for tp in [batches_to_reach(pw, ref[t])] if tp]
    return max(best) if best else savings_final(pw, ref)


def label_savings_report(gp, reg, cls):
    """How many times fewer labels the pathwise planner needs than each active learning baseline (random sampling,
    static and sequential uncertainty sampling, and the strongest of the three in each setting) to reach the same
    posterior variance.  Curves: GP = mean over the 20 draws of each dimension; Kumo-Tabular = geometric mean over
    the ten splits of each dataset of var_t / var_0.  Summaries over settings (dimension or dataset x budget): the
    maximum ("up to") and the geometric mean ("on average") of the best-stage factor, as reported in the paper; the
    geometric mean of the final-stage factor is printed for reference."""
    lines = ["== Label savings of the pathwise planner (factor = times fewer labels to reach the same variance)"]
    modules = {}
    for regime, steps in GP_BUDGETS:
        for d in GP_DIMS:
            mean = lambda key, d=d: np.mean([r[0] for r in gp[d][key].values()], 0)
            modules.setdefault("GP", []).append((f"{regime} d={d}", mean(("pathwise", steps)),
                                                 dict({b: mean((b, 0)) for b in BASELINES},
                                                      reinforce_baseline=mean(("reinforce_baseline", steps)))))
    for task, data in (("regression", reg), ("classification", cls)):
        for regime, steps in KUMO_BUDGETS:
            for lab, cell in data.items():
                rel = lambda key, cell=cell: rel_var_curve([cell], key)[0]
                modules.setdefault("Kumo-Tabular", []).append((f"{regime} {task[:3]} {lab}", rel(("pathwise", steps)),
                                                               dict({b: rel((b, 0)) for b in BASELINES},
                                                                    reinforce_baseline=rel(("reinforce_baseline", steps)))))
    for mod, rows in modules.items():
        fin, best = {}, {}
        lines.append(f"-- {mod}: final stage / best stage, against random | U static | U sequential | strongest baseline"
                     " | REINFORCE with baseline at the same budget")
        for name, pw, refs in rows:
            strongest = min(BASELINES, key=lambda b: refs[b][T])  # active learning baselines only
            refs = dict(refs, strongest=refs[strongest])
            cells = []
            for b in ["random", "uncertainty", "uncertainty_seq", "strongest", "reinforce_baseline"]:
                f, g = savings_final(pw, refs[b]), savings_best(pw, refs[b])
                fin.setdefault(b, []).append(f)
                best.setdefault(b, []).append(g)
                cells.append(f"{f:4.1f}/{g:4.1f}")
            lines.append(f"   {name:34s} " + " | ".join(cells) + f"   (strongest: {strongest})")
        for b in ["random", "uncertainty", "uncertainty_seq", "strongest"]:
            gm = lambda v: float(np.exp(np.mean(np.log(v))))
            lines.append(f"   {mod} vs {b:16s}: up to {max(best[b]):.1f}x, {gm(best[b]):.2f}x on average "
                         f"(best stage per setting; final-stage average {gm(fin[b]):.2f}x)")
        if mod == "GP":  # the paper's GP averages use the high-compute settings only
            idx = [i for i, (name, _, _) in enumerate(rows) if name.startswith("high")]
            for b in ["random", "strongest"]:
                lines.append(f"   GP vs {b:16s}: {gm([best[b][i] for i in idx]):.2f}x on average over the "
                             "high-compute settings (best stage), as reported in the paper")
        for regime in ("low", "high"):  # REINFORCE with baseline differs by budget, so report each budget
            idx = [i for i, (name, _, _) in enumerate(rows) if name.startswith(regime)]
            f = [fin["reinforce_baseline"][i] for i in idx]
            g = [best["reinforce_baseline"][i] for i in idx]
            lines.append(f"   {mod} vs REINFORCE with baseline, {regime} compute: up to {max(g):.1f}x (best stage), "
                         f"final stage {min(f):.1f}-{max(f):.1f}x")
    return lines


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_gp_d1(data):
    """Main text: d = 1, twenty function draws; one figure per compute regime."""
    for regime, steps in GP_BUDGETS:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
        for m in METHODS:
            runs = data[1][(m, 0) if m in BASELINES else (m, steps)]
            vm, vse = mean_se(runs, 0)
            em, ese = rmse_se(runs)
            plot_curve(axes[0], vm, vse, vse, m)
            plot_curve(axes[1], em, ese, ese, m)
        mean = lambda key: mean_se(data[1][key], 0)[0]
        mark_savings(axes[0], mean(("pathwise", steps)),
                     {m: mean((m, 0) if m in BASELINES else (m, steps)) for m in SAVINGS_REFS})
        style_axis(axes[0], "Var of $\\ell_2$ loss")
        style_axis(axes[1], "RMSE of estimated model $\\ell_2$ loss")
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        add_legend(fig)
        draw_savings(fig)
        save(fig, f"gp_d1_{regime}_compute.pdf")


SAVINGS_REFS = ["random", "uncertainty_seq", "reinforce_baseline"]  # references marked in the main-text figures


def best_stage(pw, ref):
    """(factor, t_ref, t_pw) at the most favourable stage of `ref` (ties: the latest stage, the longest arrow)."""
    best = (0.0, None, None)
    for t in range(1, T + 1):
        tp = batches_to_reach(pw, ref[t])
        if tp and t / tp >= best[0]:
            best = (t / tp, t, tp)
    return best


def mark_savings(ax, pw, refs, min_factor=1.5):
    """Record the label-savings arrows of one panel (drawn later by draw_savings, after the figure layout is final).
    Each arrow sits at the reference's posterior variance after t_ref batches and spans from the batch t_pw at which
    the pathwise planner first reaches that variance to t_ref; its label is t_ref / t_pw (times fewer labels)."""
    arrows = []
    for m, ref in refs.items():
        f, tr, tp = best_stage(pw, ref)
        if tr is not None and f >= min_factor:
            arrows.append({"y": float(ref[tr]), "tp": tp, "tr": tr, "f": f, "m": m})
    ax._savings = arrows


def draw_savings(fig, nudge=0.022, lw=4.2, head=26, fs=23):
    """Draw the recorded arrows of every panel.  Arrows stay at their true height; two arrows that overlap in x and
    are closer than `nudge` (fraction of the y-range) are separated by exactly `nudge`, just enough not to touch.
    Each label takes the first candidate position (above / below the middle, beside or above / below an end) that
    lies inside the panel and does not collide with another label or arrow.  lw, head and fs set the arrow width,
    arrowhead size and label font size (smaller in the appendix small multiples, which also need a larger `nudge`
    because their panels are shorter).  On a short arrow the heads shrink so that the shaft stays visible."""
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        arrows = getattr(ax, "_savings", None)
        if not arrows:
            continue
        lo, hi = ax.get_ylim()
        arrows = sorted(arrows, key=lambda a: -a["y"])
        for k in range(1, len(arrows)):
            for prev in arrows[:k]:
                overlap_x = min(prev["tr"], arrows[k]["tr"]) > max(prev["tp"], arrows[k]["tp"])
                if overlap_x and abs(prev["y"] - arrows[k]["y"]) < nudge * (hi - lo):
                    arrows[k]["y"] = prev["y"] - nudge * (hi - lo)
        boxes = []  # display-space boxes of the arrows and of the labels placed so far
        for a in arrows:
            (x0, y0), (x1, _) = ax.transData.transform([(a["tp"], a["y"]), (a["tr"], a["y"])])
            span = (x1 - x0) * 72 / fig.dpi  # arrow length in points
            ax.annotate("", xy=(a["tp"], a["y"]), xytext=(a["tr"], a["y"]), zorder=6,
                        arrowprops=dict(arrowstyle="<|-|>", color=COLOR[a["m"]], lw=lw,
                                        mutation_scale=min(head, 0.6 * span), shrinkA=0, shrinkB=0))
            boxes.append(matplotlib.transforms.Bbox([[x0, y0 - 7], [x1, y0 + 7]]))
        axbox = ax.get_window_extent(renderer)
        for n, a in enumerate(arrows):
            label = f"{a['f']:.0f}$\\times$" if abs(a["f"] - round(a["f"])) < 1e-9 else f"{a['f']:.1f}$\\times$"
            mid = ((a["tp"] + a["tr"]) / 2, a["y"])
            candidates = [(mid, (0, 6), "center", "bottom"), (mid, (0, -7), "center", "top"),
                          ((a["tr"], a["y"]), (8, 0), "left", "center"), ((a["tp"], a["y"]), (-8, 0), "right", "center"),
                          ((a["tr"], a["y"]), (0, 6), "right", "bottom"), ((a["tp"], a["y"]), (0, 6), "left", "bottom"),
                          ((a["tr"], a["y"]), (0, -7), "right", "top"), ((a["tp"], a["y"]), (0, -7), "left", "top")]
            chosen = None
            for xy, off, ha, va in candidates:
                t = ax.annotate(label, xy=xy, xytext=off, textcoords="offset points", ha=ha, va=va,
                                color=INK, fontsize=fs, zorder=7,
                                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
                bb = t.get_window_extent(renderer)
                inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
                clash = any(bb.overlaps(o) for i2, o in enumerate(boxes) if i2 != n)
                if inside and not clash:
                    chosen = t
                    break
                t.remove()
            if chosen is None:  # fall back to the first candidate
                xy, off, ha, va = candidates[0]
                chosen = ax.annotate(label, xy=xy, xytext=off, textcoords="offset points", ha=ha, va=va,
                                     color=INK, fontsize=fs, zorder=7,
                                     bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
            boxes.append(chosen.get_window_extent(renderer))


def fig_gp_d1_combined(data):
    """Main text: d = 1, twenty function draws; rows = low / high compute, columns = posterior variance, RMSE
    (each panel title names the regime and the metric); one legend."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.4))
    names = {"low": "Low compute", "high": "High compute"}
    for i, (regime, steps) in enumerate(GP_BUDGETS):
        for m in METHODS:
            runs = data[1][(m, 0) if m in BASELINES else (m, steps)]
            vm, vse = mean_se(runs, 0)
            em, ese = rmse_se(runs)
            plot_curve(axes[i, 0], vm, vse, vse, m)
            plot_curve(axes[i, 1], em, ese, ese, m)
        mean = lambda key: mean_se(data[1][key], 0)[0]
        mark_savings(axes[i, 0], mean(("pathwise", steps)),
                     {m: mean((m, 0) if m in BASELINES else (m, steps)) for m in SAVINGS_REFS})
        set_title(axes[i, 0], f"{names[regime]}: variance of $\\ell_2$ loss")
        set_title(axes[i, 1], f"{names[regime]}: RMSE of estimated $\\ell_2$ loss")
        last = i == len(GP_BUDGETS) - 1
        style_axis(axes[i, 0], "Var of $\\ell_2$ loss", xlabel=XLABEL if last else None)
        style_axis(axes[i, 1], "RMSE of estimated model $\\ell_2$ loss", xlabel=XLABEL if last else None)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.5)
    add_legend(fig)
    draw_savings(fig)
    save(fig, "gp_d1_combined.pdf")


def fig_gp_dims(data):
    """Appendix: every dimension; rows = variance, RMSE; columns = dimensions."""
    for regime, steps in GP_BUDGETS:
        fig, axes = plt.subplots(2, len(GP_DIMS), figsize=(26, 9.2))
        for j, d in enumerate(GP_DIMS):
            for m in METHODS:
                runs = data[d][(m, 0) if m in BASELINES else (m, steps)]
                vm, vse = mean_se(runs, 0)
                em, ese = rmse_se(runs)
                plot_curve(axes[0, j], vm, vse, vse, m, lw=2.6)
                plot_curve(axes[1, j], em, ese, ese, m, lw=2.6)
            mean = lambda key, d=d: mean_se(data[d][key], 0)[0]
            mark_savings(axes[0, j], mean(("pathwise", steps)),
                         {m: mean((m, 0) if m in BASELINES else (m, steps)) for m in SAVINGS_REFS})
            set_title(axes[0, j], f"$d={d}$")
            style_axis(axes[0, j], "Var of $\\ell_2$ loss" if j == 0 else None, xlabel=None, fs=17, ts=13)
            style_axis(axes[1, j], "RMSE of estimated $\\ell_2$ loss" if j == 0 else None, fs=17, ts=13)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        add_legend(fig, fs=16)
        draw_savings(fig, nudge=0.05, lw=3.2, head=20, fs=19)
        save(fig, f"gp_dims_{regime}_compute.pdf")


def fig_kumo_pooled(task, data):
    """Pooled over the datasets, one figure per compute regime (low: main text; high: appendix), in the style of the
    GP figures: left = posterior variance relative to t=0, right = RMSE relative to t=0."""
    tgt = KUMO[task]["target"]
    cells = list(data.values())
    for regime, steps in KUMO_BUDGETS:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
        for m in METHODS:
            key = (m, 0) if m in BASELINES else (m, steps)
            plot_curve(axes[0], *rel_var_curve(cells, key), m)
            plot_curve(axes[1], *rel_rmse_curve(cells, key), m)
        curve = lambda m: rel_var_curve(cells, (m, 0) if m in BASELINES else (m, steps))[0]
        mark_savings(axes[0], curve("pathwise"), {m: curve(m) for m in SAVINGS_REFS})
        style_axis(axes[0], f"Var of {tgt} (relative to $t=0$)")
        style_axis(axes[1], f"RMSE of estimated {tgt}\n(relative to $t=0$)")
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        add_legend(fig)
        draw_savings(fig)
        save(fig, f"kumo_{task}_pooled_{regime}_compute.pdf")


def fig_kumo_pooled_combined(reg, cls):
    """Pooled over the datasets, one figure per compute regime (low: main text; high: appendix); rows = regression,
    classification; columns = posterior variance and RMSE relative to t=0 (each panel title names the task and the
    metric); one legend."""
    for regime, steps in KUMO_BUDGETS:
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.4))
        for i, (task, data) in enumerate((("regression", reg), ("classification", cls))):
            tgt = KUMO[task]["target"]
            cells = list(data.values())
            for m in METHODS:
                key = (m, 0) if m in BASELINES else (m, steps)
                plot_curve(axes[i, 0], *rel_var_curve(cells, key), m)
                plot_curve(axes[i, 1], *rel_rmse_curve(cells, key), m)
            curve = lambda m: rel_var_curve(cells, (m, 0) if m in BASELINES else (m, steps))[0]
            mark_savings(axes[i, 0], curve("pathwise"), {m: curve(m) for m in SAVINGS_REFS})
            name = "Regression" if task == "regression" else "Classification"
            set_title(axes[i, 0], f"{name} (variance of {tgt})")
            set_title(axes[i, 1], f"{name} (RMSE of estimated {tgt})")
            xl = XLABEL if i == 1 else None
            style_axis(axes[i, 0], f"Var of {tgt} (relative to $t=0$)", xlabel=xl)
            style_axis(axes[i, 1], f"RMSE of estimated {tgt}\n(relative to $t=0$)", xlabel=xl)
        fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.5)
        add_legend(fig)
        draw_savings(fig)
        save(fig, f"kumo_pooled_{regime}_compute.pdf")


def fig_kumo_datasets(task, data):
    """Appendix: rows = variance (relative to t=0), RMSE; columns = datasets; one figure per regime."""
    tgt = KUMO[task]["target"]
    for regime, steps in KUMO_BUDGETS:
        fig, axes = plt.subplots(2, len(data), figsize=(22, 9.2))
        for j, (lab, cell) in enumerate(data.items()):
            for m in METHODS:
                key = (m, 0) if m in BASELINES else (m, steps)
                plot_curve(axes[0, j], *rel_var_curve([cell], key), m, lw=2.6)
                em, ese = rmse_se(cell[key])
                plot_curve(axes[1, j], em, ese, ese, m, lw=2.6)
            curve = lambda m, cell=cell: rel_var_curve([cell], (m, 0) if m in BASELINES else (m, steps))[0]
            mark_savings(axes[0, j], curve("pathwise"), {m: curve(m) for m in SAVINGS_REFS})
            set_title(axes[0, j], lab)
            style_axis(axes[0, j], f"Var of {tgt}\n(relative to $t=0$)" if j == 0 else None, xlabel=None, fs=17, ts=13)
            style_axis(axes[1, j], f"RMSE of estimated {tgt}" if j == 0 else None, fs=17, ts=13)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        add_legend(fig, fs=16)
        draw_savings(fig, nudge=0.05, lw=3.2, head=20, fs=19)
        save(fig, f"kumo_{task}_{regime}_compute.pdf")


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def table_rows(cells, steps_list, pooled_label):
    """Rows of (label, {col: (var % change, rmse ratio)}) with a pooled row (variance: all splits; RMSE: geometric
    mean of the per-setting ratios)."""
    cols = [(m, 0) for m in BASELINES[1:]] + [(m, s) for s in steps_list for m in PLANNERS]
    rows, pool_lr, pool_rr = [], {c: [] for c in cols}, {c: [] for c in cols}
    for lab, cell in cells:
        vals = {}
        for c in cols:
            lr, rr = vs_random(cell, c)
            vals[c] = (pct(lr.mean()), rr)
            pool_lr[c] += list(lr)
            pool_rr[c].append(np.log(rr))
        rows.append((lab, vals))
    rows.append((pooled_label, {c: (pct(np.mean(pool_lr[c])), float(np.exp(np.mean(pool_rr[c])))) for c in cols}))
    return cols, rows


def fmt_block(cols, rows, kind, steps_list):
    """kind 0 = variance (% change), 1 = RMSE ratio.  Bold: at each budget, the lowest value among the two uncertainty
    baselines and the three planners run at that budget (a baseline is bold if it is lowest at either budget)."""
    out = []
    for lab, vals in rows:
        shown = {c: (f"${vals[c][0]:+.0f}$" if kind == 0 else f"{vals[c][1]:.2f}") for c in cols}
        key = {c: (round(vals[c][0]) if kind == 0 else round(vals[c][1], 2)) for c in cols}
        bold = set()
        for s in steps_list:
            group = [c for c in cols if c[1] in (0, s)]
            best = min(key[c] for c in group)
            bold |= {c for c in group if key[c] == best}
        cells = []
        for c in cols:
            s = shown[c]
            if c in bold:
                s = f"$\\mathbf{{{s[1:-1]}}}$" if s.startswith("$") else f"\\textbf{{{s}}}"
            cells.append(s)
        if lab.endswith("pooled"):
            out.append("\\addlinespace[2pt]")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return out


def table_header(steps_list, first):
    n = 2 + 3 * len(steps_list)
    head = [f" & \\multicolumn{{2}}{{c}}{{Baselines}}"
            + "".join(f" & \\multicolumn{{3}}{{c}}{{{s} step{'s' if s > 1 else ''} per epoch}}" for s in steps_list) + " \\\\",
            "\\cmidrule(lr){2-3}" + "".join(f"\\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(steps_list))),
            f"{first} & U & U-seq" + " & R & Rb & PW" * len(steps_list) + " \\\\"]
    return n, head


def write_gp_table(data):
    steps = [s for _, s in GP_BUDGETS]
    cols, rows = table_rows([(f"$d={d}$", data[d]) for d in GP_DIMS], steps, "All dimensions, pooled")
    n, head = table_header(steps, "Dimension")
    lines = ["\\begin{tabular}{l" + "c" * n + "}", "\\toprule", *head, "\\midrule",
             f"\\multicolumn{{{n + 1}}}{{l}}{{\\emph{{Posterior variance of $Q$ at $t=10$, \\% change relative to random sampling}}}} \\\\",
             *fmt_block(cols, rows, 0, steps), "\\midrule",
             f"\\multicolumn{{{n + 1}}}{{l}}{{\\emph{{RMSE of the estimate of $Q$ at $t=10$ (oracle truth), ratio to random sampling}}}} \\\\",
             *fmt_block(cols, rows, 1, steps), "\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    open(os.path.join(OUT, "tables", "tab_gp_dims.tex"), "w").write("\n".join(lines) + "\n")
    print("saved tables/tab_gp_dims.tex")


# The three Kumo-Tabular tables: (file name, columns as (method, gradient steps), column headers, layout).
# Layout "stacked": the regression panel above the classification panel, labelled by rotated row labels on the
# left (task, then final variance / final RMSE), so no header rows are needed (narrow; Tables 1a and 1b stand side
# by side in the main text).  Layout "side": the two tasks next to each other under their task headings (Table 3,
# appendix), with the same rotated final variance / final RMSE labels on the left.
KUMO_TABLES = [
    ("tab_kumo_baselines", [("random", 0), ("uncertainty", 0), ("uncertainty_seq", 0), ("pathwise", 50)],
     ["Random", "U", "U-seq", "PW"], "stacked"),
    ("tab_kumo_planners_1", [("reinforce", 1), ("reinforce_baseline", 1), ("pathwise", 1)], ["R", "Rb", "PW"],
     "stacked"),
    ("tab_kumo_planners_50", [("reinforce", 50), ("reinforce_baseline", 50), ("pathwise", 50)], ["R", "Rb", "PW"],
     "side"),
]
KUMO_TASK_TITLES = [("Regression (population MSE)", "regression"), ("Classification (population log-loss)",
                                                                     "classification")]
KUMO_BLOCKS = ["Final variance (relative to $t=0$)", "Final RMSE (relative to $t=0$)"]


def rel_to_start(cell, key):
    """(final posterior variance / its value at t=0, geometric mean over splits, as per-split log ratios;
    RMSE over splits at t=10 / RMSE over splits at t=0).  t=0 is the same state for every method."""
    runs = cell[key]
    lv = [np.log(r[0][T] / r[0][0]) for r in runs.values()]
    e = np.array([r[1] for r in runs.values()])
    return lv, float(np.sqrt(e[:, T].mean() / e[:, 0].mean()))


def kumo_task_rows(data, cols):
    """For one task: [variance rows, RMSE rows], each a list of (dataset, formatted cells) ending with the pooled
    row.  Per dataset: geometric mean over the ten splits (variance) and RMSE over splits (error).  Pooled: all 40
    splits (variance) and geometric mean of the four per-dataset ratios (RMSE).  Bold: lowest value in the row."""
    # table rows name the dataset only ("MIMIC-IV"); the task is clear from the panel
    vals = {lab.replace(" LOS", "").replace(" mortality", ""): {c: rel_to_start(cell, c) for c in cols}
            for lab, cell in data.items()}
    rows_v = [(lab, [float(np.exp(np.mean(v[c][0]))) for c in cols]) for lab, v in vals.items()]
    rows_v.append(("Pooled", [float(np.exp(np.mean(sum((v[c][0] for v in vals.values()), [])))) for c in cols]))
    rows_e = [(lab, [v[c][1] for c in cols]) for lab, v in vals.items()]
    rows_e.append(("Pooled", [float(np.exp(np.mean([np.log(v[c][1]) for v in vals.values()]))) for c in cols]))
    blocks = []
    for rows in (rows_v, rows_e):
        out = []
        for lab, xs in rows:
            best = min(round(x, 2) for x in xs)
            out.append((lab, [f"\\textbf{{{x:.2f}}}" if round(x, 2) == best else f"{x:.2f}" for x in xs]))
        blocks.append(out)
    return blocks


def write_kumo_tables(reg, cls):
    """Three tables of the final posterior variance of Q and the RMSE of its estimate, both relative to their
    values at t=0 (before any label is acquired), for both tasks (see kumo_task_rows and KUMO_TABLES)."""
    data = {"regression": reg, "classification": cls}
    for name, cols, heads, layout in KUMO_TABLES:
        n = len(cols)
        tasks = [(title, kumo_task_rows(data[task], cols)) for title, task in KUMO_TASK_TITLES]
        if layout == "stacked":  # row labels on the left (rotated): task over 10 rows, metric over 5 rows
            short = {"Regression (population MSE)": ("Regression", "(MSE)"),
                     "Classification (population log-loss)": ("Classification", "(log-loss)")}
            metric = [("Final", "variance"), ("Final", "RMSE")]
            w = n + 3
            lines = ["\\begin{tabular}{ccl" + "c" * n + "}", "\\toprule",
                     " & & Dataset & " + " & ".join(heads) + " \\\\", "\\midrule"]
            for t, (title, blocks) in enumerate(tasks):
                if t > 0:
                    lines.append("\\midrule")
                for k, rows in enumerate(blocks):
                    if k > 0:
                        lines.append(f"\\cmidrule(l){{2-{w}}}")
                    for i, (lab, cells) in enumerate(rows):
                        first = second = ""
                        if k == 0 and i == 0:
                            a, b = short[title]
                            first = (f"\\multirow{{{2 * len(rows)}}}{{*}}{{\\rotatebox[origin=c]{{90}}"
                                     f"{{\\shortstack{{\\textbf{{{a}}}\\\\{b}}}}}}}")
                        if i == 0:
                            a, b = metric[k]
                            second = (f"\\multirow{{{len(rows)}}}{{*}}{{\\rotatebox[origin=c]{{90}}"
                                      f"{{\\shortstack{{\\emph{{{a}}}\\\\\\emph{{{b}}}}}}}}}")
                        if lab == "Pooled":
                            lines.append("\\addlinespace[2pt]")
                        lines.append(" & ".join([first, second, lab, *cells]) + " \\\\")
        else:  # side by side: one block of columns per task (task as column heading), rotated metric labels on the left
            w = 2 * (n + 1) + 1
            num = "w{c}{3em}" * n  # fixed width (array package), so the columns stay even under the task headings
            heading = {"Regression (population MSE)": "Regression (MSE)",
                       "Classification (population log-loss)": "Classification (log-loss)"}
            lines = ["\\begin{tabular}{cl" + num + "@{\\hspace{2em}}l" + num + "}", "\\toprule",
                     " & " + " & ".join(f"\\multicolumn{{{n + 1}}}{{c}}{{\\textbf{{{heading[title]}}}}}"
                                        for title, _ in tasks) + " \\\\",
                     f"\\cmidrule(lr){{2-{n + 2}}}\\cmidrule(lr){{{n + 3}-{w}}}",
                     " & " + " & ".join(["Dataset & " + " & ".join(heads)] * 2) + " \\\\", "\\midrule"]
            metric = [("Final", "variance"), ("Final", "RMSE")]
            for k in range(len(KUMO_BLOCKS)):
                if k > 0:
                    lines.append("\\midrule")
                pairs = list(zip(tasks[0][1][k], tasks[1][1][k]))
                for i, ((lab_r, cells_r), (lab_c, cells_c)) in enumerate(pairs):
                    first = ""
                    if i == 0:
                        a, b = metric[k]
                        first = (f"\\multirow{{{len(pairs)}}}{{*}}{{\\rotatebox[origin=c]{{90}}"
                                 f"{{\\shortstack{{\\emph{{{a}}}\\\\\\emph{{{b}}}}}}}}}")
                    if lab_r == "Pooled":
                        lines.append("\\addlinespace[2pt]")
                    lines.append(" & ".join([first, lab_r, *cells_r, lab_c, *cells_c]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(OUT, "tables", name + ".tex"), "w").write("\n".join(lines) + "\n")
        print(f"saved tables/{name}.tex")


# ----------------------------------------------------------------------------
# Numbers quoted in the text
# ----------------------------------------------------------------------------
def summary(gp, reg, cls):
    out = []
    p = out.append
    p("== GP, d = 1 (twenty draws): final variance and RMSE vs random; paired comparisons")
    for regime, steps in GP_BUDGETS:
        cell = gp[1]
        for m in METHODS:
            key = (m, 0) if m in BASELINES else (m, steps)
            lr, rr = vs_random(cell, key)
            vm = np.mean([r[0][T] for r in cell[key].values()])
            p(f"  {regime:4s} {m:18s} var(t=10)={vm:.4f}  var vs random {pct(lr.mean()):+6.1f}%  RMSE ratio {rr:.2f}")
        for other in [("reinforce_baseline", steps), ("reinforce", steps), ("uncertainty_seq", 0)]:
            m_, se, w, n = paired(cell, ("pathwise", steps), other)
            p(f"  {regime:4s} PW vs {other[0]:18s}: {pct(m_):+6.1f}% (s.e. log {se:.3f}), lower on {w}/{n}")
    p("== GP pooled over all dimensions (120 draws)")
    for regime, steps in GP_BUDGETS:
        for other in [("reinforce_baseline", steps), ("reinforce", steps), ("uncertainty_seq", 0), ("random", 0)]:
            lr = []
            for d in GP_DIMS:
                cell = gp[d]
                ks = sorted(set(cell[("pathwise", steps)]) & set(cell[other]))
                lr += list(np.log([cell[("pathwise", steps)][k][0][T] / cell[other][k][0][T] for k in ks]))
            lr = np.array(lr)
            p(f"  {regime:4s} PW vs {other[0]:18s}: {pct(lr.mean()):+6.1f}%, lower on {(lr < 0).sum()}/{len(lr)}")
        for d in GP_DIMS:
            m_, se, w, n = paired(gp[d], ("pathwise", steps), ("uncertainty_seq", 0))
            m2, se2, w2, n2 = paired(gp[d], ("pathwise", steps), ("reinforce_baseline", steps))
            p(f"     d={d:3d} PW vs U-seq {pct(m_):+6.1f}% ({w}/{n}); PW vs Rb {pct(m2):+6.1f}% ({w2}/{n2})")
    for task, data in (("regression", reg), ("classification", cls)):
        p(f"== Kumo {task}: pooled over real datasets (all splits)")
        for regime, steps in KUMO_BUDGETS:
            for other in [("reinforce_baseline", steps), ("reinforce", steps), ("random", 0), ("uncertainty", 0), ("uncertainty_seq", 0)]:
                lr = []
                for cell in data.values():
                    ks = sorted(set(cell[("pathwise", steps)]) & set(cell[other]))
                    lr += list(np.log([cell[("pathwise", steps)][k][0][T] / cell[other][k][0][T] for k in ks]))
                lr = np.array(lr)
                p(f"  {regime:4s} PW vs {other[0]:18s}: {pct(lr.mean()):+6.1f}% (s.e. log {lr.std(ddof=1) / np.sqrt(len(lr)):.3f}), lower on {(lr < 0).sum()}/{len(lr)}")
            for lab, cell in data.items():
                best = min(BASELINES, key=lambda b: vs_random(cell, (b, 0))[0].mean())
                m_, se, w, n = paired(cell, ("pathwise", steps), (best, 0))
                m2, _, w2, _ = paired(cell, ("pathwise", steps), ("reinforce_baseline", steps))
                p(f"     {lab:20s} PW vs best baseline ({best}) {pct(m_):+6.1f}% ({w}/{n}); vs Rb {pct(m2):+6.1f}% ({w2}/{n})")
        p(f"   planner wall time per decision epoch (s, mean over runs):")
        for regime, steps in KUMO_BUDGETS:
            for m in PLANNERS:
                ts = [np.mean(r[2]) for cell in data.values() for r in cell[(m, steps)].values() if r[2]]
                p(f"     {regime:4s} {m:18s} {np.mean(ts):8.1f}")
    out += label_savings_report(gp, reg, cls)
    txt = "\n".join(out)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
    print(txt)


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT, help="output folder (figures, tables/, summary.txt)")
    OUT = p.parse_args().out
    gp = load_gp()
    reg, cls = load_kumo("regression"), load_kumo("classification")
    fig_gp_d1(gp)
    fig_gp_d1_combined(gp)
    fig_gp_dims(gp)
    fig_kumo_pooled_combined(reg, cls)
    for task, data in (("regression", reg), ("classification", cls)):
        fig_kumo_pooled(task, data)
        fig_kumo_datasets(task, data)
    write_gp_table(gp)
    write_kumo_tables(reg, cls)
    summary(gp, reg, cls)


if __name__ == "__main__":
    main()
