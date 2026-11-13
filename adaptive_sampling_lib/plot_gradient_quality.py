"""Plot a gradient-quality study: expected cosine similarity to the reference gradient against the number of rollouts N.

Reads the JSON written by experiments/gp/gradient_quality.py or experiments/tabpfn/regression/gradient_quality.py.  With
several JSONs (e.g. several GP function draws) each point is the mean over the files and the band one standard error
across them; with one file, the band is the standard error over its replications.

  python -m adaptive_sampling_lib.plot_gradient_quality --json results/.../gradient_quality_dim1_ds0.json
  python -m adaptive_sampling_lib.plot_gradient_quality --json <draw 0> <draw 1> ... --taus 0.1 0.5 2.0 --name gp_dim1
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5 of the validated palette, fixed order
MARKERS = ["o", "s", "^", "D", "v"]


def series(d):
    """Estimator names of a study in display order: REINFORCE, REINFORCE-b, then pathwise by increasing tau."""
    names = {k.split("@N")[0] for k in d["cos"]}
    pw = sorted((n for n in names if n.startswith("pathwise_tau")), key=lambda n: float(n[len("pathwise_tau"):]))
    return [n for n in ("reinforce", "reinforce_baseline") if n in names] + pw


def label(name):
    if name == "reinforce":
        return "REINFORCE (no baseline)"
    if name == "reinforce_baseline":
        return "REINFORCE (mean baseline)"
    return f"Pathwise, tau = {float(name[len('pathwise_tau'):]):g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", nargs="+", required=True)
    p.add_argument("--taus", type=float, nargs="+", default=None, help="pathwise temperatures to show (default: all, at most 3)")
    p.add_argument("--name", default=None, help="output file name (default: from the first JSON)")
    p.add_argument("--out", default=None, help="output folder (default: a 'figures' folder next to the first JSON)")
    a = p.parse_args()
    ds = [json.load(open(f)) for f in a.json]
    names = [n for n in series(ds[0]) if not n.startswith("pathwise") or a.taus is None
             or any(abs(float(n[len("pathwise_tau"):]) - t) < 1e-9 for t in a.taus)]
    if len(names) > len(COLORS):
        raise SystemExit(f"{len(names)} series but {len(COLORS)} palette slots: choose at most 3 temperatures with --taus")
    Ns = ds[0]["Ns"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []  # (x, y) of each series' last point, for the end labels
    for name, color, marker in zip(names, COLORS, MARKERS):
        pts = []
        for N in Ns:
            vals = [d["cos"][f"{name}@N{N}"] for d in ds if f"{name}@N{N}" in d["cos"]]
            if len(vals) != len(ds):
                continue
            m = sum(v["mean"] for v in vals) / len(vals)
            if len(vals) > 1:
                se = math.sqrt(sum((v["mean"] - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
            else:
                se = vals[0]["se"]
            pts.append((N, m, se))
        if not pts:
            continue
        N, m, se = zip(*pts)
        ax.fill_between(N, [x - s for x, s in zip(m, se)], [x + s for x, s in zip(m, se)], color=color, alpha=0.12, linewidth=0)
        ax.plot(N, m, color=color, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, solid_capstyle="round",
                solid_joinstyle="round", label=label(name))
        ends.append((N[-1], m[-1]))
    lo, hi = ax.get_ylim()
    gap, placed = 0.05 * (hi - lo), []
    for x, y in sorted(ends, key=lambda e: (e[0], e[1])):  # end labels pushed apart where series end close together
        ty = y
        for px, py in placed:
            if px == x and ty - py < gap:
                ty = py + gap
        placed.append((x, ty))
        ax.text(x * 1.08, ty, f"{y:.2f}", va="center", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_xlim(0.85, max(Ns) * 1.5)
    ax.set_xlabel("Number of rollouts N per gradient estimate", color=INK2)
    ax.set_ylabel("Cosine similarity to reference gradient", color=INK2)
    d0 = ds[0]
    over = f", mean over {len(ds)} problems" if len(ds) > 1 else ""
    ax.set_title(f"Gradient quality, {d0.get('belief', 'TabPFN-2.5')} belief ({d0['dataset']}, pool {d0['pool']}, b = {d0['b']}{over})",
                 color=INK, fontsize=10, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.json[0])), "figures")
    name = a.name or os.path.splitext(os.path.basename(a.json[0]))[0]
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), facecolor=SURFACE)
    print("saved", os.path.join(out, f"{name}.png"))


if __name__ == "__main__":
    main()
