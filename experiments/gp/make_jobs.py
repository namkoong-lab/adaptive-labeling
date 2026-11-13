"""Job lists for the GP experiments (one line per run; the launcher skips runs whose '## <output>' file exists).

  python make_jobs.py --grid paper --out jobs.txt
  python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 1 --per_gpu 4 --log_dir logs

Grids (--grid; see README.md):
  paper   GP in R^d, d in {1, 5, 10, 20, 50, 100}, twenty function draws (dataset seeds 1-20, training seed = dataset seed):
          the three baselines; REINFORCE and REINFORCE-b at 1, 5, 10, 25 and 100 gradient steps per epoch (tags draws_*);
          the pathwise planner at tau = 0.5 at the same budgets (tags draws_tau05_*)
Without --grid, the options below build a custom grid (e.g. --extra '--tau 1.0' --tag_prefix draws_tau10 --methods pathwise,
or --seeds 0-9 for one function draw with ten training seeds).
Regimes: low = 1 step, high = --n_steps_high steps, mid<k> = k steps per epoch.  Baselines do not plan, so they are only
run in the low regime and reused for every budget.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root
from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("gp_high_dim")
BASELINES = ("random", "uncertainty", "uncertainty_seq")
DIMS = [1, 5, 10, 20, 50, 100]
BUDGETS = ["low", "mid5", "mid10", "mid25", "high"]


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def grid_jobs(dims, regimes, methods, pairs, tag_prefix, extra="", G=10, n_steps_high=100):
    """Job lines for every regime x dimension x method x (dataset seed, training seed); tag = <tag_prefix>_<regime>_compute."""
    lines = []
    for regime in regimes:
        n_steps = 1 if regime == "low" else (int(regime[3:]) if regime.startswith("mid") else n_steps_high)
        tag = f"{tag_prefix}_{regime}_compute" if tag_prefix else f"{regime}_compute"
        for d in dims:
            for meth in methods:
                if regime != "low" and meth in BASELINES:
                    continue  # baselines do not depend on the compute regime; reuse the low-compute runs
                for ds, s in pairs:
                    out = os.path.join(RESULTS_ROOT, tag, f"dim{d}", f"{meth}_ds{ds}_s{s}.json")
                    cmd = (f"{{PY}} {HERE}/run.py --method {meth} --input_dim {d} --seed {s} --seed_dataset {ds} --n_steps {n_steps} "
                           f"--G {G} --tag {tag} --gpu 0" + (f" {extra}" if extra else ""))
                    lines.append(f"{cmd} ## {out}")
    return lines


def paper_grid():
    pairs = [(k, k) for k in range(1, 21)]
    return (grid_jobs(DIMS, ["low"], list(BASELINES) + ["reinforce", "reinforce_baseline"], pairs, "draws")
            + grid_jobs(DIMS, BUDGETS[1:], ["reinforce", "reinforce_baseline"], pairs, "draws")
            + grid_jobs(DIMS, BUDGETS, ["pathwise"], pairs, "draws_tau05", extra="--tau 0.5"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=["paper"], default=None, help="a predefined grid (see the docstring); overrides the options below")
    p.add_argument("--dims", type=int, nargs="+", default=DIMS)
    p.add_argument("--dataset_seeds", type=str, default="1-20", help="one run per dataset seed k (a new GP function draw) with training seed k")
    p.add_argument("--seeds", type=str, default=None, help="instead: one function draw (dataset seed 0) and these training seeds; tags <regime>_compute")
    p.add_argument("--regimes", type=str, nargs="+", default=["low", "high"])
    p.add_argument("--methods", type=str, nargs="+", default=list(BASELINES) + ["reinforce", "reinforce_baseline", "pathwise"])
    p.add_argument("--n_steps_high", type=int, default=100)
    p.add_argument("--G", type=int, default=10, help="rollouts per gradient estimate")
    p.add_argument("--tag_prefix", type=str, default="draws", help="tag = <tag_prefix>_<regime>_compute")
    p.add_argument("--extra", type=str, default="", help="extra arguments appended to every run.py command, e.g. '--tau 0.5'")
    p.add_argument("--out", type=str, default="jobs.txt")
    args = p.parse_args()
    if args.grid == "paper":
        lines = paper_grid()
    else:
        if args.seeds is not None:  # single function draw, several training seeds (the first GP experiment)
            pairs, prefix = [(0, s) for s in parse_range(args.seeds)], ""
        else:
            pairs, prefix = [(k, k) for k in parse_range(args.dataset_seeds)], args.tag_prefix
        lines = grid_jobs(args.dims, args.regimes, args.methods, pairs, prefix, args.extra, args.G, args.n_steps_high)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {args.out}")


if __name__ == "__main__":
    main()
