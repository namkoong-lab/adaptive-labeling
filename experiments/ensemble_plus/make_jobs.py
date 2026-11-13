"""Job list for the Ensemble+ multi-step experiment (one line per run; the launcher skips runs whose '## <output>' exists).

  python make_jobs.py --grid v5 --out jobs.txt

Grids (--grid; see README.md):
  v5   21 clusters, pool 200 (the reported layout): ten seeds, baselines, and the three planners at 1 and 50 gradient steps;
       pathwise backpropagates through the unrolled training one rollout at a time (--grad_chunk 1) to bound memory
  v4   11 clusters, pool 100
Without --grid, --setting / --version / --seeds / --regimes build a custom grid.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root
from adaptive_sampling_lib.paths import results  # noqa: E402
RESULTS_ROOT = results("ensemble_multistep")


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=["v4", "v5"], default=None, help="a predefined grid; overrides --setting, --version and --pathwise_extra")
    p.add_argument("--seeds", type=str, default="0-9")
    p.add_argument("--regimes", type=str, nargs="+", default=["low", "high"])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"])
    p.add_argument("--n_steps_high", type=int, default=50)
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--setting", type=str, default="complex", choices=["complex", "complex21", "paper"])
    p.add_argument("--version", type=str, default="v4", help="tag version, e.g. v4 (11 clusters) or v5 (21 clusters)")
    p.add_argument("--pathwise_extra", type=str, default="", help="extra args appended to pathwise runs only")
    p.add_argument("--extra", type=str, default="", help="extra args appended to every run")
    p.add_argument("--out", type=str, default="jobs.txt")
    args = p.parse_args()
    if args.grid == "v5":
        args.setting, args.version, args.pathwise_extra = "complex21", "v5", "--grad_chunk 1"
    elif args.grid == "v4":
        args.setting, args.version = "complex", "v4"
    if args.setting == "complex":
        setting_args = "--layout gmm --input_dim 5 --n_anchor 11 --n_train 20 --n_pool 100 --eval_per_cluster 20 --T 5 --b 5"
    elif args.setting == "complex21":  # 21 clusters: 10 heavy (20 eval points) + 10 light (1 point) + training cluster; pool 5/cluster
        setting_args = "--layout gmm --input_dim 5 --n_anchor 21 --n_train 20 --n_pool 200 --eval_per_cluster 20 --T 5 --b 5"
    else:  # the paper's proof-of-concept scale: 4 clusters, 20 labels, pool 10, one batch of 2
        setting_args = "--layout paper --input_dim 1 --n_anchor 5 --n_train 20 --n_pool 10 --eval_per_cluster 50 --T 1 --b 2"
    lines = []
    for regime in args.regimes:
        n_steps = 1 if regime == "low" else args.n_steps_high
        tag = f"{args.setting}_{args.version}_{regime}_compute"
        for meth in args.methods:
            if regime == "high" and meth in ("random", "uncertainty", "uncertainty_seq"):
                continue
            for s in parse_range(args.seeds):
                out = os.path.join(RESULTS_ROOT, tag, f"{meth}_ds0_s{s}.json")
                cmd = (f"{{PY}} {HERE}/run.py --method {meth} --seed {s} --n_steps {n_steps} --G {args.G} {setting_args} "
                       f"--tag {tag} --gpu 0 {args.extra} {args.pathwise_extra if meth == 'pathwise' else ''}")
                lines.append(f"{cmd} ## {out}")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {args.out}")


if __name__ == "__main__":
    main()
