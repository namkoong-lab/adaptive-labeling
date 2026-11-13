"""Agreement between posterior variance and realised squared error across independent GP function draws.

For each dimension / compute regime / method: mean posterior variance at the final epoch, mean squared error of the
estimate against the oracle truth, their ratio (should be ~1 for a calibrated belief), and the error ordering.
Writes a LaTeX table and a scatter figure (sqrt mean variance vs RMSE, one point per method and dimension).

  python analyze_draws.py [--T 10] [--out report_oct2026]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/gp_high_dim"
METHODS = ["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"]
SHORT = {"random": "Random", "uncertainty": "Uncertainty (static)", "uncertainty_seq": "Uncertainty (sequential)",
         "reinforce": "REINFORCE (no baseline)", "reinforce_baseline": "REINFORCE (mean baseline)", "pathwise": "Pathwise (Autodiff)"}
COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834", "uncertainty_seq": "#1baf7a", "reinforce": "#eda100",
          "reinforce_baseline": "#e87ba4", "pathwise": "#008300"}
MARK = {1: "o", 5: "s", 10: "P", 20: "^", 50: "X", 100: "D"}

p = argparse.ArgumentParser()
p.add_argument("--T", type=int, default=10)
p.add_argument("--dims", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
p.add_argument("--regimes", nargs="+", default=["low", "mid5", "mid10", "mid25", "high"])
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_oct2026"))
a = p.parse_args()


def load(tag, d, m):
    out = []
    for f in sorted(glob.glob(f"{R}/{tag}/dim{d}/{m}_ds*_s*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = r["horizon"]
        if len(h) < a.T + 1 or "actual_target_oracle" not in h[a.T]:
            continue
        x = h[a.T]
        out.append((x["var_target"], (x["est_target"] - x["actual_target_oracle"]) ** 2, r["args"]["seed_dataset"]))
    return out


rows_tex, stats = [], {}
print(f"{'regime':5s} {'d':>4s} {'method':26s} {'n':>3s} {'mean Var':>10s} {'MSE(orc)':>10s} {'mean z^2':>10s} {'RMSE':>8s} {'sqrt(Var)':>10s}")
REG_LABEL = {"low": "Low compute (1 step / epoch)", "mid5": "Medium compute (5 steps / epoch)", "mid10": "Medium compute (10 steps / epoch)", "mid25": "Medium compute (25 steps / epoch)", "high": "High compute (100 steps / epoch)"}
for reg in a.regimes:
    for d in a.dims:
        for m in METHODS:
            tag = "draws_low_compute" if (m in METHODS[:3] or reg == "low") else f"draws_{reg}_compute"
            v = load(tag, d, m)
            if not v:
                continue
            var = np.array([x[0] for x in v]); se2 = np.array([x[1] for x in v])
            n, mv, ms = len(v), var.mean(), se2.mean()
            # calibration statistic: mean over draws of z^2 = (error^2 / posterior variance), expectation 1 under calibration
            # (equal weight per draw; the ratio of means is dominated by the draws with the largest variance)
            z2 = se2 / var
            ratio, ratio_se = z2.mean(), (z2.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
            stats[(reg, d, m)] = dict(n=n, mean_var=mv, mse=ms, ratio=ratio, ratio_se=ratio_se, ratio_of_means=ms / mv, rmse=np.sqrt(ms), sqrt_var=np.sqrt(mv))
            print(f"{reg:5s} {d:4d} {SHORT[m]:26s} {n:3d} {mv:10.4f} {ms:10.4f} {ratio:5.2f}+-{ratio_se:4.2f} {np.sqrt(ms):8.4f} {np.sqrt(mv):10.4f}")
# LaTeX table: per regime x dim block, methods as rows: sqrt(mean Var), RMSE, ratio
lines = []
for reg in a.regimes:
    label = REG_LABEL.get(reg, reg)
    if not any((reg, d, m) in stats for d in a.dims for m in METHODS):
        continue
    lines.append(f"\\multicolumn{{{1 + 3 * len(a.dims)}}}{{l}}{{\\emph{{{label}}}}}\\\\")
    for m in METHODS:
        cells = []
        for d in a.dims:
            s = stats.get((reg, d, m))
            if s is None:
                cells += ["--", "--", "--"]
                continue
            # best (lowest) RMSE and lowest sqrt var per column in bold
            best_rmse = min(stats[(reg, d, mm)]["rmse"] for mm in METHODS if (reg, d, mm) in stats)
            best_var = min(stats[(reg, d, mm)]["sqrt_var"] for mm in METHODS if (reg, d, mm) in stats)
            c1 = f"{s['sqrt_var']:.3g}"; c2 = f"{s['rmse']:.3g}"
            if s["sqrt_var"] == best_var: c1 = f"\\textbf{{{c1}}}"
            if s["rmse"] == best_rmse: c2 = f"\\textbf{{{c2}}}"
            cells += [c1, c2, f"{s['ratio']:.2f}$\\pm${s['ratio_se']:.2f}"]
        lines.append(f"{SHORT[m]} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
lines = lines[:-1]
head = "\\begin{tabular}{l" + "ccc" * len(a.dims) + "}\n\\toprule\n & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$d={d}$}}" for d in a.dims) + "\\\\\n"
head += "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(a.dims))) + "\nMethod & " + " & ".join(["$\\sqrt{\\overline{\\mathrm{Var}}}$ & RMSE & $\\overline{z^2}$"] * len(a.dims)) + "\\\\\n\\midrule\n"
os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
with open(os.path.join(a.out, "tables", "tab_draws.tex"), "w") as f:
    f.write(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
# scatter figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
regs = [r for r in a.regimes if any(k[0] == r for k in stats)]
fig, axes = plt.subplots(1, len(regs), figsize=(3.6 * len(regs), 3.4), squeeze=False); axes = axes[0]
for ax, reg in zip(axes, regs):
    xs, ys = [], []
    for (rg, d, m), s in stats.items():
        if rg != reg:
            continue
        ax.scatter(s["sqrt_var"], s["rmse"], color=COLORS[m], marker=MARK.get(d, "o"), s=34, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        xs.append(s["sqrt_var"]); ys.append(s["rmse"])
    lo, hi = min(xs + ys) * 0.8, max(xs + ys) * 1.25
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ticks = [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("root mean posterior variance (belief's predicted error)", color=INK2, fontsize=8)
    ax.set_title(REG_LABEL.get(reg, reg).replace(" compute", ""), loc="left", color=INK, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7)
axes[0].set_ylabel("RMSE vs oracle truth over function draws", color=INK2, fontsize=8)
from matplotlib.lines import Line2D
h1 = [Line2D([0], [0], marker="o", color=COLORS[m], lw=0, ms=5, label=SHORT[m]) for m in METHODS]
h2 = [Line2D([0], [0], marker=MARK[d], color=INK2, lw=0, ms=5, label=f"$d={d}$") for d in a.dims if d in MARK]
fig.legend(handles=h1 + h2, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03), labelcolor=INK2)
fig.patch.set_facecolor(SURFACE)
for ax in axes: ax.set_facecolor(SURFACE)
fig.tight_layout(rect=(0, 0.14, 1, 1))
os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, "figures", f"fig_gp_draws_agreement.{ext}"), dpi=200, facecolor=SURFACE)

def load_curves(tag, d, m):
    V, E = [], []
    for f in sorted(glob.glob(f"{R}/{tag}/dim{d}/{m}_ds*_s*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        h = r["horizon"]
        if len(h) < a.T + 1 or "actual_target_oracle" not in h[a.T]:
            continue
        V.append([x["var_target"] for x in h[: a.T + 1]]); E.append([(x["est_target"] - x["actual_target_oracle"]) ** 2 for x in h[: a.T + 1]])
    return np.array(V), np.array(E)


for reg in regs:
    fig, axes = plt.subplots(2, len(a.dims), figsize=(2.3 * len(a.dims) + 0.6, 4.6), squeeze=False)
    for j, d in enumerate(a.dims):
        for m in METHODS:
            tag = "draws_low_compute" if (m in METHODS[:3] or reg == "low") else f"draws_{reg}_compute"
            V, E = load_curves(tag, d, m)
            if len(V) == 0:
                continue
            x = np.arange(a.T + 1)
            n = len(V)
            vm, vse = V.mean(0), V.std(0, ddof=1) / np.sqrt(n) if n > 1 else 0 * V.mean(0)
            em = np.sqrt(E.mean(0)); ese = (E.std(0, ddof=1) / np.sqrt(n)) / np.maximum(2 * em, 1e-300) if n > 1 else 0 * em
            for ax, mean, se in ((axes[0, j], vm, vse), (axes[1, j], em, ese)):
                ax.fill_between(x, mean - se, mean + se, color=COLORS[m], alpha=0.12, lw=0)
                ax.plot(x, mean, color=COLORS[m], lw=1.5, marker="o", ms=2.8, mec=SURFACE, mew=0.5)
        axes[0, j].set_title(f"$d={d}$", loc="left", color=INK, fontsize=9)
        axes[1, j].set_xlabel("decision epoch $t$", color=INK2, fontsize=8)
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(True, axis="y", color=GRID, lw=0.6)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7)
            ax.set_facecolor(SURFACE)
    axes[0, 0].set_ylabel("posterior variance of MSE", color=INK2, fontsize=8)
    axes[1, 0].set_ylabel("RMSE of estimate vs oracle truth", color=INK2, fontsize=8)
    fig.legend(handles=[Line2D([0], [0], color=COLORS[m], lw=1.8, label=SHORT[m]) for m in METHODS], loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01), labelcolor=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, "figures", f"fig_gp_draws_curves_{reg}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

# final values against dimension, averaged over draws: sqrt(mean Var) and RMSE, both regimes
fig, axes = plt.subplots(2, len(regs), figsize=(3.4 * len(regs), 5.2), sharex=True, squeeze=False)
for j, reg in enumerate(regs):
    for m in METHODS:
        ds = [d for d in a.dims if (reg, d, m) in stats]
        if not ds: continue
        axes[0, j].plot(ds, [stats[(reg, d, m)]["sqrt_var"] for d in ds], color=COLORS[m], lw=1.6, marker="o", ms=3.5, mec=SURFACE, mew=0.6)
        axes[1, j].errorbar(ds, [stats[(reg, d, m)]["rmse"] for d in ds], yerr=[stats[(reg, d, m)]["rmse"] * 0 for d in ds], color=COLORS[m], lw=1.6, marker="o", ms=3.5, mec=SURFACE, mew=0.6)
    axes[0, j].set_title(f"{REG_LABEL.get(reg, reg).split(' (')[0]}: forecast $\\sqrt{{\\overline{{\\mathrm{{Var}}}}}}$, $t=10$", loc="left", color=INK, fontsize=8)
    axes[1, j].set_title(f"{REG_LABEL.get(reg, reg).split(' (')[0]}: realised RMSE, $t=10$", loc="left", color=INK, fontsize=8)
    for ax in axes[:, j]:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(a.dims); ax.set_xticklabels([str(d) for d in a.dims]); ax.minorticks_off()
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=7); ax.set_facecolor(SURFACE)
    axes[1, j].set_xlabel("input dimension $d$", color=INK2, fontsize=8)
fig.legend(handles=[Line2D([0], [0], color=COLORS[m], lw=1.8, marker="o", ms=4, label=SHORT[m]) for m in METHODS], loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01), labelcolor=INK2)
fig.patch.set_facecolor(SURFACE); fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, "figures", f"fig_gp_draws_dims.{ext}"), dpi=200, facecolor=SURFACE)
plt.close(fig)
print("wrote tab_draws.tex, fig_gp_draws_agreement, fig_gp_draws_curves_{low,high}, fig_gp_draws_dims")
