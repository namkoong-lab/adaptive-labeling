"""Job lists for the Kumo-Tabular experiments (one line per run; the launcher skips runs whose '## <output>' file exists).

  python make_jobs.py --task regression --grid final --out jobs.txt
  python -m adaptive_sampling_lib.launch jobs.txt --gpus 0 1 --per_gpu 3 --log_dir logs

Grids (--grid; see README.md):
  final   the TabPFN final grid (same splits, datasets, budgets and tau = 0.5) with Kumo-Tabular medium as the belief and the
          fractional update the paper reports: every context row attends to itself at full weight (--kumo_self_attn full)
          and, for regression, the pseudo-updates use the labelled data's preprocessing statistics (--kumo_fixed_stats both;
          the default of the classification runner) and range-conditioned moments.  Tags final_kumo_sf_* (regression) and
          final_kumo_cls_sf_* (classification).
Without --grid, the options of ../tabpfn/make_jobs.py build a custom grid (pass the Kumo flags in --extra).
"""
import argparse
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tabpfn_make_jobs", os.path.join(HERE, "..", "tabpfn", "make_jobs.py"))
tj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tj)  # the same grid logic and dataset settings as the TabPFN experiments
FINAL = {"regression": ("final_kumo_sf", "--moments range --tau 0.5 --kumo_fixed_stats both --kumo_self_attn full"),
         "classification": ("final_kumo_cls_sf", "--tau 0.5 --kumo_self_attn full")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--grid", choices=["final"], default=None, help="a predefined grid (see the docstring); overrides the options below")
    p.add_argument("--datasets", nargs="+", default=None, help="default: the four real datasets and the synthetic one of the task")
    p.add_argument("--dataset_seeds", default="1-10")
    p.add_argument("--regimes", nargs="+", default=["low", "mid10", "high"])
    p.add_argument("--n_steps_high", type=int, default=50)
    p.add_argument("--methods", nargs="+", default=tj.METHODS)
    p.add_argument("--tag_prefix", default="custom")
    p.add_argument("--extra", default="", help="extra arguments appended to every run.py command (e.g. the --kumo_* flags)")
    p.add_argument("--out", default="jobs.txt")
    a = p.parse_args()
    run_py = os.path.join(HERE, a.task, "run.py")
    datasets = a.datasets or tj.REAL[a.task] + [tj.SYNTHETIC[a.task]]
    if a.grid == "final":
        tag_prefix, extra = FINAL[a.task]
        lines = tj.grid_jobs(run_py, a.task, tj.REAL[a.task] + [tj.SYNTHETIC[a.task]], [(k, k) for k in range(1, 11)], ["low", "mid10", "high"],
                             tj.METHODS, tag_prefix, extra)
    else:
        lines = tj.grid_jobs(run_py, a.task, datasets, [(k, k) for k in tj.parse_range(a.dataset_seeds)], a.regimes, a.methods,
                             a.tag_prefix, a.extra, a.n_steps_high)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} jobs to {a.out}")


if __name__ == "__main__":
    main()
