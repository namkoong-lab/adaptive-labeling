# Results of the paper's experiments

The figures, tables and per-run results behind the experiments of *Adaptive Measurement via Posterior Variance
Minimization*.  Everything here can be redrawn from `runs/` without rerunning any experiment.

## Figures and tables

| File | Content |
|---|---|
| `gp_d1_combined.pdf` (`gp_d1_{low,high}_compute.pdf`: one regime each) | GP, d = 1, twenty function draws; 1 and 100 gradient steps per epoch |
| `gp_dims_{low,high}_compute.pdf`, `tables/tab_gp_dims.tex` | GP in every dimension d in {1, 5, 10, 20, 50, 100} |
| `cosine_similarity_vs_N.pdf` | gradient quality: cosine similarity with the true gradient against the rollouts per estimate N |
| `gp_rollouts_vs_N.pdf` | planning with N = 1, 2, 10 rollouts per gradient estimate (GP, d = 1) |
| `gp_eicu_{low,high}_compute.pdf` | GP on eICU, 1 and 1000 gradient steps per epoch |
| `kumo_pooled_{low,high}_compute.pdf` (and one task per figure: `kumo_{regression,classification}_pooled_*`) | Kumo-Tabular, pooled over the real datasets |
| `kumo_{regression,classification}_{low,high}_compute.pdf` | Kumo-Tabular, each dataset |
| `tables/tab_kumo_{baselines,planners_1,planners_50}.tex` | Kumo-Tabular: the pathwise planner against the baselines; the three planners at 1 and 50 steps |
| `summary.txt` | every number quoted in the text (label savings, paired comparisons, timings) |

## Per-run results (`runs/`)

One JSON per run, in the layout `<experiment>/<tag>/<problem>/<method>_ds<split>_s<seed>.json` written by the
experiments (`experiments/*/run.py`).  These are the runs of the grids `make_jobs.py --grid ...` that the main README
lists, and of the gradient-quality study (`gp_high_dim/gradient_quality/`).  Fields of a run:

- `method`, `b` (batch size), `T` (epochs), `seed`, `args` (all settings of the run), `problem` (set sizes, dataset),
  `uq` (the belief), `planner_kwargs`;
- `horizon`: one entry per epoch t = 0 ... T with `var_target` (posterior variance of the target), `est_target`
  (its posterior mean, the estimate), `actual_target` (the target on the evaluation set's labels), `abs_err` and `sq_err`
  (error of the estimate) and, for synthetic data, `actual_target_oracle` with `abs_err_oracle` and `sq_err_oracle`
  (against the denoised value of the generating GP);
- `selected`: the batch labelled at each epoch, as positions in the pool at that epoch (labelled points leave the
  pool); `selected_clusters`: their clusters, where the problem has clusters;
- `timing`: seconds per epoch; `total_time`.

The per-gradient-step logs of the planners and machine-specific paths are left out (`paper_figures/export_runs.py`
writes these copies from a full results folder).

## Redrawing

```bash
ADAPTIVE_SAMPLING_RESULTS=paper_results/runs python paper_figures/make_figures.py --out paper_results
ADAPTIVE_SAMPLING_RESULTS=paper_results/runs python paper_figures/make_gp_extra_figures.py --out paper_results
```
