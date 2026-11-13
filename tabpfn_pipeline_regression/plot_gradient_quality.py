"""Plot the TabPFN gradient-quality study (cosine to the reference gradient vs number of rollouts N).

  python plot_gradient_quality.py [--json <path>] [--out <dir>]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--json", default="/shared/share_mala/daksh/adaptive_sampling_experiments/results/tabpfn_regression/gradient_quality_eicu.json")
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
a = p.parse_args()
d = json.load(open(a.json))

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [  # fixed categorical order (validated palette)
    ("reinforce", "REINFORCE (no baseline)", "#2a78d6", "o"),
    ("reinforce_baseline", "REINFORCE (mean baseline)", "#eb6834", "s"),
    ("pathwise_tau0.1", "Pathwise, tau = 0.1", "#1baf7a", "^"),
    ("pathwise_tau0.5", "Pathwise, tau = 0.5", "#eda100", "D"),
    ("pathwise_tau2.0", "Pathwise, tau = 2", "#e87ba4", "v"),
]
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for key, label, color, marker in SERIES:
    pts = sorted((int(k.split("@N")[1]), v["mean"], v["se"]) for k, v in d["cos"].items() if k.split("@N")[0] == key)
    if not pts:
        continue
    N, m, se = zip(*pts)
    ax.fill_between(N, [x - s for x, s in zip(m, se)], [x + s for x, s in zip(m, se)], color=color, alpha=0.12, linewidth=0)
    ax.plot(N, m, color=color, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, solid_capstyle="round", solid_joinstyle="round", label=label)
    ax.annotate(f"{m[-1]:.2f}", (N[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
ax.axhline(0, color=AXIS, lw=1, zorder=0)
ax.set_xscale("log")
ax.set_xticks(d["Ns"])
ax.set_xticklabels([str(n) for n in d["Ns"]])
ax.minorticks_off()
ax.set_xlim(0.85, 150)
ax.set_xlabel("Number of rollouts N per gradient estimate", color=INK2)
ax.set_ylabel("Cosine similarity to reference gradient", color=INK2)
ax.set_title(f"Gradient quality with TabPFN-2.5 as UQ module ({d['dataset']}, pool {d['pool']}, b = {d['b']})", color=INK, fontsize=10, loc="left")
ax.grid(True, axis="y", color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
leg = ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"gradient_quality_{d['dataset']}.{ext}"), facecolor=SURFACE)
print("saved", os.path.join(a.out, f"gradient_quality_{d['dataset']}.png"))
