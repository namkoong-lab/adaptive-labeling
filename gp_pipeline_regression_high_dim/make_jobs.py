"""Generate the job list for experiment 1 (high-dim GP).

  python make_jobs.py --dims 1 5 10 20 50 100 --seeds 0-9 --regime low high --out jobs.txt
Then
  python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 4 7 --per_gpu 4 --log_dir logs
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = "/shared/share_mala/daksh/adaptive_sampling_experiments/results/gp_high_dim"


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
    p.add_argument("--dims", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--seeds", type=str, default="0-9")
    p.add_argument("--regimes", type=str, nargs="+", default=["low", "high"])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["random", "uncertainty", "uncertainty_seq", "reinforce", "reinforce_baseline", "pathwise"])
    p.add_argument("--n_steps_high", type=int, default=100)
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--out", type=str, default="jobs.txt")
    p.add_argument("--dataset_seeds", type=str, default=None,
                   help="if given, one run per dataset seed k (new GP function draw) with training seed k; tag prefix 'draws'")
    p.add_argument("--tag_prefix", type=str, default="draws", help="with --dataset_seeds: tag = <tag_prefix>_<regime>_compute")
    p.add_argument("--extra", type=str, default="", help="extra arguments appended to every run.py command, e.g. '--tau 0.5'")
    args = p.parse_args()
    lines = []
    for regime in args.regimes:
        # regimes: low = 1 step, high = n_steps_high steps, mid<k> = k steps per epoch (planners only)
        n_steps = 1 if regime == "low" else (int(regime[3:]) if regime.startswith("mid") else args.n_steps_high)
        tag = f"{regime}_compute" if args.dataset_seeds is None else f"{args.tag_prefix}_{regime}_compute"
        for d in args.dims:
            for meth in args.methods:
                if regime != "low" and meth in ("random", "uncertainty", "uncertainty_seq"):
                    continue  # baselines do not depend on compute regime; reuse the low-compute runs
                if args.dataset_seeds is None:
                    pairs = [(0, s) for s in parse_range(args.seeds)]
                else:
                    pairs = [(k, k) for k in parse_range(args.dataset_seeds)]
                for ds, s in pairs:
                    out = os.path.join(RESULTS_ROOT, tag, f"dim{d}", f"{meth}_ds{ds}_s{s}.json")
                    cmd = (f"{{PY}} {HERE}/run.py --method {meth} --input_dim {d} --seed {s} --seed_dataset {ds} --n_steps {n_steps} "
                           f"--G {args.G} --tag {tag} --gpu 0" + (f" {args.extra}" if args.extra else ""))
                    lines.append(f"{cmd} ## {out}")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {args.out}")


if __name__ == "__main__":
    main()
