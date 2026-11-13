"""Job lists for the TabPFN experiments (one line per run; the launcher skips runs whose '## <output>' file exists).

  python make_jobs.py --task regression --grid final --out jobs.txt
  python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 1 --per_gpu 3 --log_dir logs

Grids (--grid; see README.md):
  final   ten dataset splits (seeds 1-10) of each real dataset and ten draws of the synthetic one; the three baselines and the
          three planners at 1, 10 and 50 gradient steps per epoch; tau = 0.5, float32, raw input with the constant-feature fix;
          regression: range-conditioned moments (tags final_rng_*); classification: tags final_cls_*
Without --grid, --datasets / --dataset_seeds / --regimes / --methods / --tag_prefix / --extra build a custom grid.
Real datasets always get the skewed pool and evaluation set (--layout skewed); the synthetic classification task uses a
latent GP of variance 25 (--outputscale 25.0).
"""
import argparse
import importlib.util
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root
BASELINES = ("random", "uncertainty", "uncertainty_seq")
METHODS = list(BASELINES) + ["reinforce", "reinforce_baseline", "pathwise"]
REAL = {"regression": ["eicu_raw", "california", "bike", "mimic_los"], "classification": ["eicu", "magic", "credit", "mimic"]}
SYNTHETIC = {"regression": "synthetic_gmm", "classification": "synthetic_logit"}
FINAL = {"regression": ("final_rng", "--moments range --input_transform fix --fp32 --tau 0.5"),
         "classification": ("final_cls", "--input_transform fix --fp32 --tau 0.5")}


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_run(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.dirname(path)) + "_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset_args(task, ds, extra):
    if ds in REAL[task]:
        return f"--layout skewed {extra}".strip()
    if ds == "synthetic_logit":
        return f"{extra} --outputscale 25.0".strip()
    return extra


def grid_jobs(run_py, task, datasets, pairs, regimes, methods, tag_prefix, extra, n_steps_high=50):
    """Job lines for every regime x dataset x method x (dataset split, training seed); tag = <tag_prefix>_<regime>.
    The output marker is computed by the runner's own output_path, so it always matches where the run writes."""
    run = load_run(run_py)
    parser = argparse.ArgumentParser()
    run.add_args(parser)
    lines = []
    for regime in regimes:
        n_steps = 1 if regime == "low" else (int(regime[3:]) if regime.startswith("mid") else n_steps_high)  # mid<k>: k steps
        tag = f"{tag_prefix}_{regime}"
        for ds in datasets:
            for m in methods:
                if regime != "low" and m in BASELINES:
                    continue  # baselines do not depend on the compute regime; reuse the low-compute runs
                for k, s in pairs:
                    a = f"--dataset {ds} --method {m} --seed {s} --seed_dataset {k} --n_steps {n_steps} --tag {tag} --gpu 0 {dataset_args(task, ds, extra)}".strip()
                    out = run.output_path(parser.parse_args(shlex.split(a)))
                    lines.append(f"{{PY}} {run_py} {a} ## {out}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--grid", choices=["final"], default=None, help="a predefined grid (see the docstring); overrides the options below")
    p.add_argument("--datasets", nargs="+", default=None, help="default: the four real datasets and the synthetic one of the task")
    p.add_argument("--dataset_seeds", default="1-10", help="one run per dataset split k (new training cluster, evaluation set, predictor) with training seed k")
    p.add_argument("--regimes", nargs="+", default=["low", "mid10", "high"])
    p.add_argument("--n_steps_high", type=int, default=50)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--tag_prefix", default="custom")
    p.add_argument("--extra", default="", help="extra arguments appended to every run.py command")
    p.add_argument("--out", default="jobs.txt")
    a = p.parse_args()
    run_py = os.path.join(HERE, a.task, "run.py")
    datasets = a.datasets or REAL[a.task] + [SYNTHETIC[a.task]]
    pairs = [(k, k) for k in parse_range(a.dataset_seeds)]
    if a.grid == "final":
        tag_prefix, extra = FINAL[a.task]
        lines = grid_jobs(run_py, a.task, REAL[a.task] + [SYNTHETIC[a.task]], [(k, k) for k in range(1, 11)], ["low", "mid10", "high"], METHODS, tag_prefix, extra)
    else:
        lines = grid_jobs(run_py, a.task, datasets, pairs, a.regimes, a.methods, a.tag_prefix, a.extra, a.n_steps_high)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {a.out}")


if __name__ == "__main__":
    main()
