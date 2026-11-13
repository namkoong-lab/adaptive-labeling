"""Standalone cosine-similarity panel from grad_diag_*.csv, styled to match the paper figure.

Colours follow errors_l2_loss_normal_low_compute: AUTODIFF green, REINFORCE-with-baseline
orange, REINFORCE-without-baseline brown. The three AUTODIFF temperatures share the paper's
green (tau=0.5 is the exact paper colour, tau=0.1 lighter, tau=10 darker) and are separated
additionally by line style and marker, so they stay distinguishable without relying on hue.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_GREEN = "#2ca02c"   # AUTODIFF in the paper
PAPER_ORANGE = "#ff7f0e"  # REINFORCE (with baseline)
PAPER_BROWN = "#8c564b"   # REINFORCE (without baseline)

# (csv name, legend label, colour, linestyle, marker)
SERIES = [
    ("relaxed_tau=0.1",  r"AUTODIFF ($\tau = 0.1$)",        "#7fd67f",    "--",  "s"),
    ("relaxed_tau=0.5",  r"AUTODIFF ($\tau = 0.5$)",        PAPER_GREEN,  "-",   "o"),
    ("relaxed_tau=10.0", r"AUTODIFF ($\tau = 10.0$)",       "#14601a",    ":",   "^"),
    ("pg_no_baseline",   "REINFORCE (without baseline)",    PAPER_BROWN,  "-.",  "v"),
    ("pg_with_baseline", "REINFORCE (with baseline)",       PAPER_ORANGE, (0, (5, 1)), "D"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="grad_diag_k5_dim1_taus.csv")
    p.add_argument("--out", default="cosine_similarity_vs_N")
    p.add_argument("--fontsize", type=float, default=18.0)
    args = p.parse_args()

    data = defaultdict(dict)
    order = []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            G = int(row["G"])
            if G not in order:
                order.append(G)
            data[row["estimator"]][G] = (float(row["cos_true"]), float(row["cos_true_se"]))

    G_list = sorted(order)
    xpos = range(len(G_list))          # equal spacing, labels carry the real values

    fs = args.fontsize
    fig, ax = plt.subplots(figsize=(10.2, 6.6))

    for key, label, color, ls, marker in SERIES:
        ys = [data[key][g][0] for g in G_list]
        es = [data[key][g][1] for g in G_list]
        ax.errorbar(list(xpos), ys, yerr=es, label=label, color=color, linestyle=ls,
                    marker=marker, markersize=9, linewidth=2.4, capsize=4,
                    markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xticks(list(xpos))
    ax.set_xticklabels([str(g) for g in G_list], fontsize=fs)
    ax.set_xlabel("N", fontsize=fs + 4)
    ax.set_ylabel("Cosine Similarity", fontsize=fs + 2)
    ax.tick_params(axis="both", labelsize=fs)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])   # no tick above 1.0; the headroom stays blank
    ax.grid(True, color="#dddddd", linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):   # full box, as in the paper figure
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("#333333")
    # headroom so the two-column legend clears the tau=10 curve, as in the paper figure
    ax.set_ylim(top=1.34)
    ax.legend(fontsize=fs - 1, frameon=True, framealpha=0.95, loc="upper left",
              ncol=2, columnspacing=1.1, handlelength=2.4, borderaxespad=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white")
        print(f"[saved] {args.out}.{ext}")


if __name__ == "__main__":
    main()
