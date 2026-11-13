"""Copy the result JSONs of the documented grids into paper_results/runs/, compact and without machine-specific paths.

The runs are exactly those of the make_jobs.py presets listed in GRIDS (plus the gradient-quality study), so the folder
holds every run behind the paper's figures and tables and nothing else.  Each copy keeps the per-epoch metrics, the
selected batches, the settings and the timings; it drops the per-gradient-step planner log ('planner_history', a
diagnostic no figure uses) and any absolute path.  The folder layout is that of the results root, so

  ADAPTIVE_SAMPLING_RESULTS=paper_results/runs python paper_figures/make_figures.py

redraws every figure and table from the copies.

  python paper_figures/export_runs.py            (reads $ADAPTIVE_SAMPLING_RESULTS, writes paper_results/runs)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, REPO)
from adaptive_sampling_lib.paths import RESULTS_ROOT  # noqa: E402

# (job generator, arguments): the grids whose runs are exported
GRIDS = [
    ("experiments/gp/make_jobs.py", ["--grid", "paper"]),
    ("experiments/gp/make_jobs.py", ["--grid", "eicu"]),
    ("experiments/gp/make_jobs.py", ["--grid", "rollouts"]),
    ("experiments/kumo/make_jobs.py", ["--task", "regression", "--grid", "final"]),
    ("experiments/kumo/make_jobs.py", ["--task", "classification", "--grid", "final"]),
    # not in the paper: the TabPFN and Ensemble+ grids of this branch
    ("experiments/tabpfn/make_jobs.py", ["--task", "regression", "--grid", "final"]),
    ("experiments/tabpfn/make_jobs.py", ["--task", "classification", "--grid", "final"]),
    ("experiments/ensemble_plus/make_jobs.py", ["--grid", "v5"]),
    ("experiments/ensemble_plus/make_jobs.py", ["--grid", "v4"]),
]
EXTRA = ["gp_high_dim/gradient_quality/gradient_quality_dim1_ds*.json"]  # results not produced by a job list


def grid_outputs(script, args):
    """The '## <output>' markers of a job list."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        subprocess.run([sys.executable, os.path.join(REPO, script), *args, "--out", tmp.name], check=True, capture_output=True)
        return [line.split(" ## ")[1].strip() for line in open(tmp.name) if " ## " in line]


def strip_paths(x):
    """Drop every string that is an absolute path (output folders, data locations)."""
    if isinstance(x, dict):
        return {k: strip_paths(v) for k, v in x.items() if not (isinstance(v, str) and v.startswith("/"))}
    if isinstance(x, list):
        return [strip_paths(v) for v in x]
    return x


def has_path(x):
    if isinstance(x, dict):
        return any(has_path(v) for v in x.values())
    if isinstance(x, list):
        return any(has_path(v) for v in x)
    return isinstance(x, str) and x.startswith("/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=RESULTS_ROOT, help="results root to read")
    p.add_argument("--out", default=os.path.join(REPO, "paper_results", "runs"))
    a = p.parse_args()
    src = os.path.realpath(a.src)
    paths = []
    for script, args in GRIDS:
        outs = grid_outputs(script, args)
        paths += [os.path.relpath(o, RESULTS_ROOT) for o in outs]
        print(f"{script} {' '.join(args)}: {len(outs)} runs")
    for pat in EXTRA:
        paths += [os.path.relpath(f, src) for f in sorted(glob.glob(os.path.join(src, pat)))]
    paths = sorted(set(paths))
    missing = [r for r in paths if not os.path.exists(os.path.join(src, r))]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing under {src}, e.g. {missing[:3]}")
    total = 0
    for rel in paths:
        r = json.load(open(os.path.join(src, rel)))  # symlinked runs are read through and written as files
        r.pop("planner_history", None)
        r = strip_paths(r)
        assert not has_path(r), rel
        s = json.dumps(r, separators=(",", ":"))
        dst = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(s + "\n")
        total += len(s) + 1
    print(f"wrote {len(paths)} runs ({total / 1e6:.1f} MB) to {a.out}")


if __name__ == "__main__":
    main()
